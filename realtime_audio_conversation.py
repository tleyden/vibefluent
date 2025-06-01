import asyncio
import json
import base64
import websockets
import pyaudio
import threading
import queue
import os
from typing import List, Optional
from onboarding import OnboardingData
from models import ConversationResponse
from database import get_database
from prompt_manager import get_prompt_manager
import logfire
from llm_agent_factory import LLMAgentFactory
from constants import DEFAULT_REALTIME_AUDIO_MODEL, MODE


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

        logfire.info(
            f"RealtimeAudioConversationAgent initialized for {self.onboarding_data.name}",
            onboarding_data=onboarding_data,
        )

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
        url = f"wss://api.openai.com/v1/realtime?model={DEFAULT_REALTIME_AUDIO_MODEL}"
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
                    "threshold": 0.85,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 400,
                },
                "tools": [
                    {
                        "type": "function",
                        "name": "record_mistake",
                        "description": "Record a language mistake made by the user during conversation",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "mistake_text": {
                                    "type": "string",
                                    "description": "The incorrect text or phrase the user said",
                                },
                                "correct_text": {
                                    "type": "string",
                                    "description": "The correct way to say it in the target language",
                                },
                                "mistake_type": {
                                    "type": "string",
                                    "enum": [
                                        "grammar",
                                        "pronunciation",
                                        "vocabulary",
                                        "word_order",
                                        "conjugation",
                                    ],
                                    "description": "The type of mistake made",
                                },
                                "explanation": {
                                    "type": "string",
                                    "description": "Brief explanation of why it's incorrect and how to fix it",
                                },
                            },
                            "required": [
                                "mistake_text",
                                "correct_text",
                                "mistake_type",
                                "explanation",
                            ],
                        },
                    }
                ],
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

        # Start in its own thread to avoid blocking the main async loop
        # which needs to send websocket messages, etc
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
            logfire.debug(
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
            logfire.debug(f"Cleared {cleared_count} audio chunks from queue")

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

        # Start in its own thread to avoid blocking the main async loop
        # which needs to send websocket messages, etc
        threading.Thread(target=audio_output_thread, daemon=True).start()

    async def _handle_websocket_messages(self):
        """Handle incoming messages from OpenAI Realtime API."""
        while self.websocket and not self.websocket.closed:
            try:
                message = await self.websocket.recv()
                data = json.loads(message)
                logfire.trace(
                    "Received WebSocket message",
                    data=data,
                )

                await self._process_websocket_message(data)

            except websockets.exceptions.ConnectionClosed as e:
                raise RuntimeError("WebSocket connection closed unexpectedly") from e
            except Exception as e:
                logfire.error(f"WebSocket message handling error: {e}.  Retrying...")

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
                print(
                    "\033[1;32mNew vocabulary words saved: "
                    + ", ".join(
                        str(word)
                        for word in vocab_response.vocab_words_user_asked_about
                    )
                    + "\033[0m"
                )

        except Exception as e:
            logfire.error(f"Error processing user transcript for vocab: {e}")

    async def _record_mistake(self, mistake_data: dict):
        """Record a language mistake made by the user."""
        try:
            # Save mistake to database
            self.db.save_mistake(
                mistake_text=mistake_data["mistake_text"],
                correct_text=mistake_data["correct_text"],
                mistake_type=mistake_data["mistake_type"],
                explanation=mistake_data["explanation"],
                native_language=self.onboarding_data.native_language,
                target_language=self.onboarding_data.target_language,
                user_id=getattr(self.onboarding_data, "user_id", None),
            )

            logfire.info(
                f"Mistake recorded for {self.onboarding_data.name}",
                mistake_data=mistake_data,
                onboarding_data=self.onboarding_data,
            )

            print(
                f"\033[1;33m🔧 Mistake recorded: '{mistake_data['mistake_text']}' → '{mistake_data['correct_text']}'\033[0m"
            )

        except Exception as e:
            logfire.error(f"Error recording mistake: {e}")

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
            logfire.debug(f"Response started: {self.response_id}")

        elif message_type == "response.done":
            # Response generation completed
            self.has_active_response = False
            self.response_id = None
            self.is_assistant_speaking = False
            logfire.debug("Response completed")

        elif message_type == "response.output_item.done":
            # Audio output item completed
            output_item = data.get("item", {})
            if output_item.get("type") == "message":
                self.is_assistant_speaking = False
                logfire.debug("Audio output completed")

        elif message_type == "input_audio_buffer.speech_started":
            logfire.debug("User started speaking")
            # Always interrupt - clear audio immediately and try to cancel if possible
            await self._interrupt_assistant()

        elif message_type == "input_audio_buffer.speech_stopped":
            logfire.debug("User stopped speaking")

        elif message_type == "conversation.item.input_audio_transcription.completed":
            # User's speech was transcribed
            transcript = data.get("transcript", "")
            if transcript:
                self.conversation_history.append(f"User: {transcript}")
                logfire.info(f"User transcript: {transcript}")

                # Add transcript to pending list for processing after assistant responds
                self.pending_user_transcripts.append(transcript)

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
                logfire.debug(f"Cancellation attempt failed (expected): {error_msg}")
            else:
                logfire.error(f"API error: {error_msg}")

        elif message_type == "response.function_call_arguments.delta":
            # Function call arguments being streamed
            pass

        elif message_type == "response.function_call_arguments.done":
            # Function call arguments complete - handle tool calls
            item = data.get("item", {})
            if item.get("type") == "function_call":
                function_name = item.get("name")
                if function_name == "record_mistake":
                    try:
                        arguments = json.loads(item.get("arguments", "{}"))
                        await self._record_mistake(arguments)

                        # Send function call result back to the API
                        result_message = {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "function_call_output",
                                "call_id": item.get("call_id"),
                                "output": json.dumps(
                                    {
                                        "success": True,
                                        "message": "Mistake recorded successfully",
                                    }
                                ),
                            },
                        }
                        await self.websocket.send(json.dumps(result_message))

                        # Trigger response generation to continue conversation
                        response_message = {"type": "response.create"}
                        await self.websocket.send(json.dumps(response_message))

                    except Exception as e:
                        logfire.error(
                            f"Error processing record_mistake function call: {e}"
                        )

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

        # Kick off the conversation
        await self.send_text_message("Greet the user and start the conversation.")

        # Start handling WebSocket messages - this blocks indefinitely
        logfire.info("Starting to handle WebSocket messages")
        await self._handle_websocket_messages()
        logfire.info("Finished handling WebSocket messages")

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


def run_realtime_audio_loop(conversation_agent: RealtimeAudioConversationAgent):
    """Run the realtime audio conversation loop."""

    async def audio_loop():
        # Start the realtime conversation
        # Note this will block indefinitely until conversation is stopped, but no way to stop it
        logfire.info(
            f"Starting realtime audio conversation for {conversation_agent.onboarding_data.name}"
        )
        success = await conversation_agent.start_conversation()
        logfire.info(
            f"Finished realtime audio conversation for {conversation_agent.onboarding_data.name}.  Result: {success}"
        )

        if not success:
            raise RuntimeError(
                "Failed to start realtime audio mode. Please check your internet connection and API key."
            )

    # Run the async audio loop
    asyncio.run(audio_loop())


def run_realtime_conversation_loop(onboarding_data):
    print("\n" + "=" * 60)
    print("🌍 Welcome to your conversation practice! 🌍")
    print(f"Mode: {MODE}")
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 're-onboard' to update your profile settings")
    print("Press Ctrl+C to exit anytime")
    print("=" * 60 + "\n")

    conversation_agent = RealtimeAudioConversationAgent(onboarding_data)
    return run_realtime_audio_loop(conversation_agent)
