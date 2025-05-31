import asyncio
import json
import base64
import websockets
import pyaudio
import threading
import queue
import os
import random
from typing import List, Optional
from onboarding import OnboardingData
from models import ConversationResponse, VocabWord
from database import get_database
from prompt_manager import get_prompt_manager
import logfire
from llm_agent_factory import LLMAgentFactory
from constants import DO_VOCAB_DRILLS


class RealtimeAudioConversationAgent:
    def __init__(self, onboarding_data: OnboardingData):
        self.onboarding_data = onboarding_data
        self.conversation_history: List[str] = []
        self.max_history = 100
        self.db = get_database()
        self.prompt_manager = get_prompt_manager()
        self.factory = LLMAgentFactory()

        # Create vocabulary extraction agent
        vocab_extraction_prompt = self.prompt_manager.render_vocab_extraction_prompt(
            self.onboarding_data
        )
        self.vocab_extraction_system_prompt = vocab_extraction_prompt
        self.vocab_extractor = self.factory.create_agent(
            result_type=ConversationResponse,
            system_prompt=vocab_extraction_prompt,
        )

        # Audio settings
        self.sample_rate = 24000
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        self.channels = 1

        # WebSocket and audio state
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.audio = pyaudio.PyAudio()
        self.is_recording = False
        self.is_playing = False
        self.audio_queue: queue.Queue = queue.Queue()
        self.response_queue: queue.Queue = queue.Queue()

        # Track response state for interruption
        self.has_active_response = False
        self.response_id = None
        self.is_assistant_speaking = (
            False  # Track if assistant is currently outputting audio
        )

        # OpenAI API key
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for realtime audio mode"
            )

        # Track pending user transcripts for vocabulary processing
        self.pending_user_transcripts: List[str] = []

        # Vocabulary drill system
        self.pending_drills: List[dict] = []  # Queue of drills to execute
        self.current_drill: Optional[dict] = None  # Currently active drill
        self.is_in_drill_mode = False
        self.initial_drills_started = False  # Track if we've started initial drills

        logfire.info(
            f"RealtimeAudioConversationAgent initialized for {self.onboarding_data.name}",
            onboarding_data=onboarding_data,
        )

    def generate_initial_question(self) -> str:
        """Generate a personalized question for realtime audio mode."""
        return f"Hello {self.onboarding_data.name}! Welcome to realtime audio mode. I'll speak with you to help practice your {self.onboarding_data.target_language}. Start speaking, and I'll respond with audio."

    def _create_system_message(self) -> str:
        """Create the system message for the OpenAI Realtime API using template."""
        vocab_words = self.db.get_all_vocab_words(self.onboarding_data)
        vocab_context = ""
        if vocab_words:
            vocab_list = [
                f"{w.word_in_target_language} ({w.word_in_native_language})"
                for w in vocab_words[:20]
            ]
            vocab_context = f"\nVocabulary words to practice: {', '.join(vocab_list)}"

        return self.prompt_manager.render_realtime_session_config(
            self.onboarding_data, vocab_context
        )

    async def _connect_websocket(self):
        """Connect to OpenAI Realtime API via WebSocket."""
        url = (
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
        )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        try:
            self.websocket = await websockets.connect(url, extra_headers=headers)
            logfire.info(
                f"Connected to OpenAI Realtime API for {self.onboarding_data.name}"
            )

            # Send session configuration
            await self._configure_session()
            return True
        except Exception as e:
            logfire.error(f"Failed to connect to OpenAI Realtime API: {e}")
            return False

    async def _configure_session(self):
        """Configure the OpenAI Realtime session."""
        # Map common language names to ISO codes for Whisper
        language_mapping = {
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Italian": "it",
            "Portuguese": "pt",
            "Russian": "ru",
            "Japanese": "ja",
            "Korean": "ko",
            "Chinese": "zh",
            "Mandarin": "zh",
            "Dutch": "nl",
            "Polish": "pl",
            "Turkish": "tr",
            "Arabic": "ar",
            "Hindi": "hi",
            "English": "en",
        }

        # Get the ISO language code for the target language
        target_lang_code = language_mapping.get(
            self.onboarding_data.target_language, "en"
        )

        config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": self._create_system_message(),
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1",
                    "language": target_lang_code,
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 400,
                },
                "tools": [],
                "tool_choice": "auto",
                "temperature": 0.8,
                "max_response_output_tokens": 4096,
            },
        }

        if not self.websocket or self.websocket.closed:
            raise RuntimeError(
                "WebSocket connection is not open. Cannot configure session."
            )

        await self.websocket.send(json.dumps(config))
        logfire.info(
            f"Session configured for realtime audio with {self.onboarding_data.target_language} transcription"
        )

    def _start_audio_input_stream(self):
        """Start recording audio from microphone."""

        def audio_input_thread():
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
            )

            while self.is_recording:
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    if self.websocket and not self.websocket.closed:
                        asyncio.run_coroutine_threadsafe(
                            self._send_audio_chunk(data), self.loop
                        )
                except Exception as e:
                    logfire.error(f"Audio input error: {e}")
                    break

            stream.stop_stream()
            stream.close()

        threading.Thread(target=audio_input_thread, daemon=True).start()

    async def _send_audio_chunk(self, audio_data):
        """Send audio chunk to OpenAI Realtime API."""
        if self.websocket and not self.websocket.closed:
            audio_b64 = base64.b64encode(audio_data).decode()
            message = {"type": "input_audio_buffer.append", "audio": audio_b64}
            await self.websocket.send(json.dumps(message))

    async def _interrupt_assistant(self):
        """Interrupt the assistant's current response when user starts speaking."""
        # Clear audio queue immediately regardless of response state
        self._clear_audio_queue()

        # Only send cancellation if there's actually an active response
        if self.websocket and not self.websocket.closed and self.has_active_response:
            interrupt_message = {"type": "response.cancel"}
            await self.websocket.send(json.dumps(interrupt_message))
            logfire.info("Interrupted assistant response due to user speech")
        else:
            logfire.trace(
                "User started speaking - cleared audio queue (no active response to cancel)"
            )

    def _clear_audio_queue(self):
        """Clear all pending audio from the queue to stop playback immediately."""
        cleared_count = 0
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                cleared_count += 1
            except queue.Empty:
                break
        if cleared_count > 0:
            logfire.trace(f"Cleared {cleared_count} audio chunks from queue")

    def _start_audio_output_stream(self):
        """Start playing audio from the response queue."""

        def audio_output_thread():
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size,
            )

            while self.is_playing:
                try:
                    if not self.audio_queue.empty():
                        audio_data = self.audio_queue.get(timeout=0.1)
                        stream.write(audio_data)
                except queue.Empty:
                    continue
                except Exception as e:
                    logfire.error(f"Audio output error: {e}")
                    break

            stream.stop_stream()
            stream.close()

        threading.Thread(target=audio_output_thread, daemon=True).start()

    async def _handle_websocket_messages(self):
        """Handle incoming messages from OpenAI Realtime API."""
        while self.websocket and not self.websocket.closed:
            try:
                message = await self.websocket.recv()
                data = json.loads(message)

                await self._process_websocket_message(data)

            except websockets.exceptions.ConnectionClosed:
                logfire.info("WebSocket connection closed")
                break
            except Exception as e:
                logfire.error(f"WebSocket message handling error: {e}")

    def _generate_vocab_drill(self, vocab_word: VocabWord) -> dict:
        """Generate a vocabulary drill based on a detected word."""
        drill_types = ["translation", "definition", "context", "reverse_translation"]
        drill_type = random.choice(drill_types)

        # Generate drill based on type and language level
        level = self.onboarding_data.target_language_level.lower()
        target_lang = self.onboarding_data.target_language
        native_lang = self.onboarding_data.native_language

        if drill_type == "translation":
            # How do you say [native word] in target language?
            question = f"How do you say '{vocab_word.word_in_native_language}' in {target_lang}?"
            expected_answer = vocab_word.word_in_target_language

        elif drill_type == "definition":
            # What does [target word] mean in native language?
            question = f"What does '{vocab_word.word_in_target_language}' mean in {native_lang}?"
            expected_answer = vocab_word.word_in_native_language

        elif drill_type == "context":
            # Use the word [target word] in a sentence
            if level == "beginner":
                question = f"Can you use the word '{vocab_word.word_in_target_language}' in a simple sentence?"
            else:
                question = f"Try using '{vocab_word.word_in_target_language}' in a sentence to show you understand it."
            expected_answer = (
                f"Any sentence containing '{vocab_word.word_in_target_language}'"
            )

        elif drill_type == "reverse_translation":
            # What's the native language word for [target word]?
            question = f"What's the {native_lang} word for '{vocab_word.word_in_target_language}'?"
            expected_answer = vocab_word.word_in_native_language

        return {
            "type": drill_type,
            "question": question,
            "expected_answer": expected_answer,
            "vocab_word": vocab_word,
        }

    async def _start_initial_vocab_drills(self):
        """Start vocabulary drills with all existing words when drills are enabled."""
        if not DO_VOCAB_DRILLS or self.initial_drills_started:
            return

        # Get all existing vocabulary words
        existing_vocab_words = self.db.get_all_vocab_words(self.onboarding_data)

        if not existing_vocab_words:
            logfire.info(f"No existing vocabulary words found for {self.onboarding_data.name}")
            return

        logfire.info(
            f"Starting initial vocabulary drills for {self.onboarding_data.name} with {len(existing_vocab_words)} words",
            vocab_words=[str(word) for word in existing_vocab_words],
        )

        # Generate drills for all existing vocabulary words
        new_drills = []
        for vocab_word in existing_vocab_words:
            drill = self._generate_vocab_drill(vocab_word)
            new_drills.append(drill)

        # Shuffle for variety
        random.shuffle(new_drills)

        # Add to pending drills queue
        self.pending_drills.extend(new_drills)

        # Start drill sequence
        self.is_in_drill_mode = True
        self.initial_drills_started = True

        # Send instructions about the drills
        await self._send_drill_instructions(len(new_drills))

        # Start first drill
        if self.pending_drills:
            self.current_drill = self.pending_drills.pop(0)
            await self._send_vocab_drill(self.current_drill)

    async def _send_drill_instructions(self, num_drills: int):
        """Send instructions about the upcoming vocabulary drills."""
        if self.initial_drills_started and not hasattr(self, '_sent_initial_instructions'):
            instructions = f"Welcome! I found {num_drills} vocabulary word{'s' if num_drills > 1 else ''} in your learning profile. Let's practice {'them' if num_drills > 1 else 'it'} with some quick drills before we start our conversation. I'll ask you a question about each word - just answer naturally!"
            self._sent_initial_instructions = True
        else:
            instructions = f"Great! I detected {num_drills} vocabulary word{'s' if num_drills > 1 else ''} you asked about. Let's practice {'them' if num_drills > 1 else 'it'} with some quick drills. I'll ask you a question about each word - just answer naturally!"

        drill_message = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": instructions}],
            },
        }

        await self.websocket.send(json.dumps(drill_message))

        # Trigger response generation
        response_message = {"type": "response.create"}
        await self.websocket.send(json.dumps(response_message))

        logfire.info(
            f"Drill instructions sent for {self.onboarding_data.name}",
            instructions=instructions,
            num_drills=num_drills,
        )

    async def _send_vocab_drill(self, drill: dict):
        """Send a vocabulary drill via conversation.item.create."""
        if not self.websocket or self.websocket.closed:
            return

        drill_question = drill["question"]

        # Create conversation item with the drill
        drill_message = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": drill_question}],
            },
        }

        await self.websocket.send(json.dumps(drill_message))

        # Trigger response generation
        response_message = {"type": "response.create"}
        await self.websocket.send(json.dumps(response_message))

        logfire.info(
            f"Vocabulary drill sent for {self.onboarding_data.name}",
            drill_question=drill_question,
            drill_type=drill["type"],
            vocab_word=str(drill["vocab_word"]),
        )

    async def _send_drill_completion_message(self):
        """Send a message when all drills are completed."""
        if self.initial_drills_started and not hasattr(self, '_completed_initial_drills'):
            completion_message = "Excellent work! You've completed all your vocabulary practice drills. Those words should stick better now. Now let's have a natural conversation - feel free to ask me about any new words you'd like to learn!"
            self._completed_initial_drills = True
        else:
            completion_message = "Excellent work! You've completed all the vocabulary drills. Those words should stick better now. Let's continue our conversation!"

        drill_message = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": completion_message}],
            },
        }

        await self.websocket.send(json.dumps(drill_message))

        # Trigger response generation
        response_message = {"type": "response.create"}
        await self.websocket.send(json.dumps(response_message))

        logfire.info(
            f"Drill completion message sent for {self.onboarding_data.name}",
            completion_message=completion_message,
        )

    async def _process_user_transcript(
        self,
        user_transcripts: List[str],
        assistant_response: str,
        conversation_history: List[str],
    ):
        """Process user transcripts with assistant response context to detect and save vocabulary words."""
        if not user_transcripts or not any(t.strip() for t in user_transcripts):
            return

        try:
            # Get last 5 messages from conversation history for context
            recent_history = (
                conversation_history[-5:]
                if len(conversation_history) > 5
                else conversation_history
            )

            # Combine user transcripts into a single context
            user_input = " ".join(user_transcripts)

            # Use prompt manager to render the realtime vocab extraction prompt
            prompt = self.prompt_manager.render_realtime_vocab_extraction_prompt(
                user_transcripts=user_input,
                assistant_response=assistant_response,
                recent_conversation_history=recent_history,
                onboarding_data=self.onboarding_data,
            )

            # Use async version instead of sync
            result = await self.vocab_extractor.run(prompt)

            logfire.info(
                f"Vocab extractor agent result for {self.onboarding_data.name}",
                result=result.data,
                prompt=prompt,
                system_prompt=self.vocab_extraction_system_prompt,
                onboarding_data=self.onboarding_data,
            )

            vocab_response = result.data

            # Save any vocabulary words that were detected
            if vocab_response.vocab_words_user_asked_about:
                self.db.save_vocab_words(
                    vocab_response.vocab_words_user_asked_about,
                    self.onboarding_data.native_language,
                    self.onboarding_data.target_language,
                )

                logfire.info(
                    f"New vocabulary words saved from audio conversation: {', '.join(str(word) for word in vocab_response.vocab_words_user_asked_about)}",
                    onboarding_data=self.onboarding_data,
                    vocab_words=vocab_response.vocab_words_user_asked_about,
                    user_transcripts=user_transcripts,
                    assistant_response=assistant_response,
                )

                # Only generate and send drills for new words if the feature is enabled and we're not already in drill mode
                if DO_VOCAB_DRILLS and not self.is_in_drill_mode:
                    logfire.info(
                        f"Generating drills for newly detected vocabulary words: {', '.join(str(word) for word in vocab_response.vocab_words_user_asked_about)}",
                        onboarding_data=self.onboarding_data,
                    )
                    # Generate drills for detected vocabulary words
                    new_drills = []
                    for vocab_word in vocab_response.vocab_words_user_asked_about:
                        drill = self._generate_vocab_drill(vocab_word)
                        new_drills.append(drill)

                    # Add to pending drills queue
                    self.pending_drills.extend(new_drills)

                    # Start drill sequence
                    self.is_in_drill_mode = True

                    # Send instructions about the drills
                    await self._send_drill_instructions(len(new_drills))

                    # Start first drill
                    self.current_drill = self.pending_drills.pop(0)
                    await self._send_vocab_drill(self.current_drill)

        except Exception as e:
            logfire.error(f"Error processing user transcript for vocab: {e}")

    async def _process_websocket_message(self, data):
        """Process different types of messages from the API."""
        message_type = data.get("type")

        if message_type == "response.audio.delta":
            # Received audio chunk - only queue if we're not being interrupted
            audio_b64 = data.get("delta")
            if audio_b64:
                audio_data = base64.b64decode(audio_b64)
                self.audio_queue.put(audio_data)
                self.is_assistant_speaking = True

        elif message_type == "response.audio_transcript.delta":
            # Received transcript chunk
            transcript = data.get("delta", "")
            if transcript:
                # Store partial transcript (could display this in real-time)
                pass

        elif message_type == "response.audio_transcript.done":
            # Complete assistant transcript received
            transcript = data.get("transcript", "")
            if transcript:
                self.conversation_history.append(f"Assistant: {transcript}")
                logfire.info(f"Assistant transcript: {transcript}")

                # Process vocabulary if we have pending user transcripts
                if self.pending_user_transcripts:
                    await self._process_user_transcript(
                        self.pending_user_transcripts,
                        transcript,
                        self.conversation_history,
                    )
                    # Reset the pending transcripts after processing
                    self.pending_user_transcripts = []

        elif message_type == "response.created":
            # Response generation started
            self.has_active_response = True
            self.response_id = data.get("response", {}).get("id")
            self.is_assistant_speaking = False
            logfire.trace(f"Response started: {self.response_id}")

        elif message_type == "response.done":
            # Response generation completed
            self.has_active_response = False
            self.response_id = None
            self.is_assistant_speaking = False
            logfire.trace("Response completed")

        elif message_type == "response.output_item.done":
            # Audio output item completed
            output_item = data.get("item", {})
            if output_item.get("type") == "message":
                self.is_assistant_speaking = False
                logfire.trace("Audio output completed")

        elif message_type == "input_audio_buffer.speech_started":
            logfire.trace("User started speaking")
            # Always interrupt - clear audio immediately and try to cancel if possible
            await self._interrupt_assistant()

        elif message_type == "input_audio_buffer.speech_stopped":
            logfire.trace("User stopped speaking")

        elif message_type == "conversation.item.input_audio_transcription.completed":
            # User's speech was transcribed
            transcript = data.get("transcript", "")
            if transcript:
                self.conversation_history.append(f"User: {transcript}")
                logfire.info(f"User transcript: {transcript}")

                # If drills are enabled and in drill mode, process drill response
                if DO_VOCAB_DRILLS and self.is_in_drill_mode and self.current_drill:
                    logfire.info(f"Processing drill response: {transcript}")

                    # Move to next drill or complete drill sequence
                    if self.pending_drills:
                        self.current_drill = self.pending_drills.pop(0)
                        await self._send_vocab_drill(self.current_drill)
                    else:
                        # All drills completed
                        self.is_in_drill_mode = False
                        self.current_drill = None
                        await self._send_drill_completion_message()
                else:
                    # Add transcript to pending list for processing after assistant responds
                    self.pending_user_transcripts.append(transcript)

                # Start initial vocab drills after first user interaction if not already started
                if DO_VOCAB_DRILLS and not self.initial_drills_started and not self.is_in_drill_mode:
                    await self._start_initial_vocab_drills()

        elif message_type == "response.cancelled":
            # Assistant response was cancelled (due to interruption)
            logfire.info("Assistant response was cancelled")
            self.has_active_response = False
            self.response_id = None
            self.is_assistant_speaking = False
            # Clear any remaining audio in the queue to stop playback immediately
            self._clear_audio_queue()
            # Clear pending transcripts since response was cancelled
            self.pending_user_transcripts = []

        elif message_type == "error":
            error_msg = data.get("error", {}).get("message", "Unknown error")
            # Don't log cancellation errors as errors - they're expected during interruption
            if "Cancellation failed" in error_msg:
                logfire.trace(f"Cancellation attempt failed (expected): {error_msg}")
            else:
                logfire.error(f"API error: {error_msg}")

    async def start_conversation(self):
        """Start the realtime audio conversation."""
        if not await self._connect_websocket():
            return False

        # Store event loop for thread communication
        self.loop = asyncio.get_event_loop()

        # Start audio streams
        self.is_recording = True
        self.is_playing = True
        self._start_audio_input_stream()
        self._start_audio_output_stream()

        # Start handling WebSocket messages
        await self._handle_websocket_messages()

        return True

    async def stop_conversation(self):
        """Stop the realtime audio conversation."""
        self.is_recording = False
        self.is_playing = False

        if self.websocket and not self.websocket.closed:
            await self.websocket.close()

        logfire.info("Realtime audio conversation stopped")

    def add_to_history(self, user_message: str):
        """Add user message to conversation history, maintaining max limit."""
        self.conversation_history.append(user_message)
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history :]

    def get_response(self, user_message: str) -> ConversationResponse:
        """Get AI response to user message in realtime audio mode."""
        # For compatibility with text mode interface
        # In actual realtime mode, this won't be used much
        self.add_to_history(user_message)

        return ConversationResponse(
            assistant_message="In realtime audio mode - responses are provided via voice.",
            follow_up_question="Continue speaking to practice your conversation skills!",
            vocab_words_user_asked_about=[],
        )

    async def send_text_message(self, message: str):
        """Send a text message to the conversation (useful for commands)."""
        if self.websocket is not None and not self.websocket.closed:
            text_message = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": message}],
                },
            }
            await self.websocket.send(json.dumps(text_message))

            # Trigger response generation
            response_message = {"type": "response.create"}
            await self.websocket.send(json.dumps(response_message))

    def __del__(self):
        """Cleanup audio resources."""
        if hasattr(self, "audio"):
            self.audio.terminate()
