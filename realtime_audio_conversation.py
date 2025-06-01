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
from models import VocabExtractResponse
from database import get_database
from prompt_manager import get_prompt_manager
import logfire
from llm_agent_factory import LLMAgentFactory
from constants import DEFAULT_REALTIME_AUDIO_MODEL, MODE, DEFAULT_REALTIME_AUDIO_VOICE


class RealtimeAudioConversationAgent:
    def __init__(self, onboarding_data: OnboardingData):
        self.onboarding_data: OnboardingData = onboarding_data
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
            result_type=VocabExtractResponse,
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
                for w in vocab_words
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
                "voice": DEFAULT_REALTIME_AUDIO_VOICE,
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
                    "silence_duration_ms": 1200,
                },
                "tools": [
                    {
                        "type": "function",
                        "name": "user_used_other_language_mistake",
                        "description": f"""
                        This function will be called when the user responds with any words in another language rather than the target language {self.onboarding_data.target_language}
                        """,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "mistake_explanation": {
                                    "type": "string",
                                    "description": f"""
                                        Brief explanation of the mistake, including the words or phrases that 
                                        were supposed to be in the target language {self.onboarding_data.target_language} 
                                        but were in a different language.
                                    """,
                                },
                            },
                            "required": [
                                "mistake_explanation",
                            ],
                        },
                    },
                    {
                        "type": "function",
                        "name": "user_asked_for_translation",
                        "description": f"""
                            If the user explicitly asks for the translation of a word 
                            into {self.onboarding_data.target_language}, this tool will save 
                            that word for future drills.
                        """,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "words_needing_translation": {
                                    "type": "string",
                                    "description": f"The word or words that the user asked to have translated into {self.onboarding_data.target_language}",
                                },
                            },
                            "required": [
                                "words_needing_translation",
                            ],
                        },
                    },
                    {
                        "type": "function",
                        "name": "user_asked_to_save_vocab_word",
                        "description": """
                            If the user explicitly asks to save a word or phrase for vocabulary drills 
                            this tool will save that word or phrase for future drills.
                        """,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "word_or_phrase_to_save": {
                                    "type": "string",
                                    "description": "The word or words that the user asked to save",
                                },
                            },
                            "required": [
                                "word_or_phrase_to_save",
                            ],
                        },
                    },
                    {
                        "type": "function",
                        "name": "user_wants_vocab_drill",
                        "description": """
                            If the user explicitly asks to do vocabulary drills or practice  
                            this tool will start the vocuabulary drill mode with the user's 
                            vocabulary words.
                        """,
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                    },
                    {
                        "type": "function",
                        "name": "vocab_drill_result",
                        "description": """
                            This function will record the result of a vocabulary drill for a particular word
                            that is being practiced.    
                        """,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "drill_word_target_language": {
                                    "type": "string",
                                    "description": f"The drill word in {self.onboarding_data.target_language}",
                                },
                            },
                            "required": ["drill_word_target_language"],
                        },
                    },
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

                await self._process_websocket_message(data)

            except websockets.exceptions.ConnectionClosed as e:
                raise RuntimeError("WebSocket connection closed unexpectedly") from e
            except Exception as e:
                logfire.error(f"WebSocket message handling error: {e}.  Retrying...")

    async def _extract_vocabulary_from_llm_response(self, llm_response: str):
        try:
            # Extract vocabulary words from the mistake explanation
            prompt = await self.prompt_manager.render_realtime_vocab_extraction_prompt(
                llm_response=llm_response,
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

            # Save to database asynchronously to avoid blocking the event loop
            vocab_response = result.data
            if vocab_response.vocab_words:
                await self.db.save_vocab_words_async(
                    vocab_response.vocab_words,
                    self.onboarding_data.native_language,
                    self.onboarding_data.target_language,
                )

                logfire.info(
                    f"New vocabulary words saved from llm response: {', '.join(str(word) for word in vocab_response.vocab_words)}",
                    onboarding_data=self.onboarding_data,
                    vocab_words=vocab_response.vocab_words,
                    mistake_explanation=llm_response,
                )
                print(
                    "\033[1;32m🫙 New vocabulary words saved: "
                    + ", ".join(str(word) for word in vocab_response.vocab_words)
                    + "\033[0m"
                )

        except Exception as e:
            logfire.exception(
                f"Error extracting vocab from llm response: {e}",
                llm_response=llm_response,
            )

    async def _user_asked_for_translation(self, llm_response: str) -> str:
        try:
            logfire.info(
                f"User asked for translation: {llm_response}",
                onboarding_data=self.onboarding_data,
            )

            print(
                f"\033[1;33m🔧 User asked for translation: '{llm_response}'.  Vocab will be extracted and saved"
            )

            return (
                f"Translate the words into {self.onboarding_data.target_language} as requested by user: {llm_response}. "
                + "Make sure that you mention that you have made a note for future practice. "
                + self._keep_conversation_going()
            )

        except Exception as e:
            logfire.error(f"Error processing translation request: {e}")
            return self._keep_conversation_going()

    async def _user_asked_to_save_vocab_word(self, llm_response: str) -> str:
        try:
            logfire.info(
                f"User asked to save vocab word: {llm_response}",
                onboarding_data=self.onboarding_data,
            )

            print(
                f"\033[1;33m🔧 User asked to save vocab word: '{llm_response}'.  Vocab will be saved for future practice"
            )

            return (
                f"Tell the user you saved the word or words in '{llm_response}' for future practice. "
                + self._keep_conversation_going()
            )

        except Exception as e:
            logfire.error(f"Error processing save vocab request: {e}")
            return self._keep_conversation_going()

    def _keep_conversation_going(self) -> str:
        return (
            f"Keep the conversation going in the target language: {self.onboarding_data.target_language}. "
            + f"Focus on the user's interests: {self.onboarding_data.conversation_interests} and language learning goals: {self.onboarding_data.reason_for_learning}."
        )

    async def _user_used_other_language_mistake(self, mistake_explanation: str) -> str:
        try:
            logfire.info(
                f"Mistake recorded for {self.onboarding_data.name}: {mistake_explanation}",
                onboarding_data=self.onboarding_data,
            )

            print(
                f"\033[1;33m🔧 Mistake encountered: '{mistake_explanation}'.  Vocab will be extracted and saved"
            )

            return (
                f"Explain the mistake to the user {mistake_explanation} in "
                + f"their native language {self.onboarding_data.native_language}.  Make sure you mention "
                + "that you have made a note for future practice.  "
                + self._keep_conversation_going()
            )

        except Exception as e:
            logfire.error(f"Error recording mistake: {e}")
            return self._keep_conversation_going()

    async def _process_websocket_message(self, data):
        """Process different types of messages from the API."""
        message_type = data.get("type")

        # Map message types to their handler methods
        handler_map = {
            "response.audio.delta": self._handle_response_audio_delta,
            "response.audio_transcript.delta": self._handle_response_audio_transcript_delta,
            "response.audio_transcript.done": self._handle_response_audio_transcript_done,
            "response.created": self._handle_response_created,
            "response.done": self._handle_response_done,
            "response.output_item.done": self._handle_response_output_item_done,
            "input_audio_buffer.speech_started": self._handle_input_audio_buffer_speech_started,
            "input_audio_buffer.speech_stopped": self._handle_input_audio_buffer_speech_stopped,
            "conversation.item.input_audio_transcription.completed": self._handle_conversation_item_input_audio_transcription_completed,
            "response.cancelled": self._handle_response_cancelled,
            "error": self._handle_error,
            "response.function_call_arguments.delta": self._handle_response_function_call_arguments_delta,
            "response.function_call_arguments.done": self._handle_response_function_call_arguments_done,
        }

        handler = handler_map.get(message_type)
        if handler:
            await handler(data)
        else:
            logfire.debug(f"Unhandled message type: {message_type}")

    async def _handle_response_audio_delta(self, data):
        """Handle response.audio.delta messages."""
        # Received audio chunk - only queue if we're not being interrupted
        audio_b64 = data.get("delta")
        if audio_b64:
            audio_data = base64.b64decode(audio_b64)
            self.audio_queue.put(audio_data)
            self.is_assistant_speaking = True

    async def _handle_response_audio_transcript_delta(self, data):
        """Handle response.audio_transcript.delta messages."""
        # Received transcript chunk
        transcript = data.get("delta", "")
        if transcript:
            # Store partial transcript (could display this in real-time)
            pass

    async def _handle_response_audio_transcript_done(self, data):
        """Handle response.audio_transcript.done messages."""
        # Complete assistant transcript received
        transcript = data.get("transcript", "")
        if transcript:
            self.conversation_history.append(f"Assistant: {transcript}")
            logfire.info(f"Assistant transcript: {transcript}")

    async def _handle_response_created(self, data):
        """Handle response.created messages."""
        # Response generation started
        self.has_active_response = True
        self.response_id = data.get("response", {}).get("id")
        self.is_assistant_speaking = False
        logfire.debug(f"Response started: {self.response_id}")

    async def _handle_response_done(self, data):
        """Handle response.done messages."""
        # Response generation completed
        self.has_active_response = False
        self.response_id = None
        self.is_assistant_speaking = False
        logfire.debug("Response completed")

    async def _handle_response_output_item_done(self, data):
        """Handle response.output_item.done messages."""
        # Audio output item completed
        output_item = data.get("item", {})
        if output_item.get("type") == "message":
            self.is_assistant_speaking = False
            logfire.debug("Audio output completed")

    async def _handle_input_audio_buffer_speech_started(self, data):
        """Handle input_audio_buffer.speech_started messages."""
        logfire.debug("User started speaking")
        # Always interrupt - clear audio immediately and try to cancel if possible
        await self._interrupt_assistant()

    async def _handle_input_audio_buffer_speech_stopped(self, data):
        """Handle input_audio_buffer.speech_stopped messages."""
        logfire.debug("User stopped speaking")

    async def _handle_conversation_item_input_audio_transcription_completed(self, data):
        """Handle conversation.item.input_audio_transcription.completed messages."""
        # User's speech was transcribed
        transcript = data.get("transcript", "")
        if transcript:
            self.conversation_history.append(f"User: {transcript}")
            logfire.info(f"User transcript: {transcript}")

            # Add transcript to pending list for processing after assistant responds
            self.pending_user_transcripts.append(transcript)

    async def _handle_response_cancelled(self, data):
        """Handle response.cancelled messages."""
        # Assistant response was cancelled (due to interruption)
        logfire.info("Assistant response was cancelled")
        self.has_active_response = False
        self.response_id = None
        self.is_assistant_speaking = False
        # Clear any remaining audio in the queue to stop playback immediately
        self._clear_audio_queue()
        # Clear pending transcripts since response was cancelled
        self.pending_user_transcripts = []

    async def _handle_error(self, data):
        """Handle error messages."""
        error_msg = data.get("error", {}).get("message", "Unknown error")
        # Don't log cancellation errors as errors - they're expected during interruption
        if "Cancellation failed" in error_msg:
            logfire.debug(f"Cancellation attempt failed (expected): {error_msg}")
        else:
            logfire.error(f"API error: {error_msg}")

    async def _handle_response_function_call_arguments_delta(self, data):
        """Handle response.function_call_arguments.delta messages."""
        # Function call arguments being streamed
        pass

    async def _handle_response_function_call_arguments_done(self, data):
        """Handle response.function_call_arguments.done messages."""
        logfire.info(
            "Received function call arguments (final output) - processing function call",
            data=data,
        )
        await self.process_function_call(data)

    async def process_function_call(self, data):
        try:
            vocab_function_calls = [
                "user_used_other_language_mistake",
                "user_asked_for_translation",
                "user_asked_to_save_vocab_word",
            ]

            logfire.info("processing function call", data=data)
            # Function call arguments complete - handle tool calls
            function_name = data.get("name", "")
            logfire.info(f"function name: {function_name}")

            if function_name in vocab_function_calls:
                # Process vocabulary-related function calls
                await self.process_vocab_function_call(function_name, data)
            elif function_name == "vocab_drill_result":
                logfire.info(
                    "Processing vocab drill result function call",
                    data=data,
                )
                # TODO: implement vocab drill result processing and respond to keep conversation going

            elif function_name == "user_wants_vocab_drill":
                vocab_words = self.db.get_all_vocab_words(self.onboarding_data)
                message = f"""
                The user wants to do vocabulary drills.  Run the user through a vocabulary drill session with the following words:

                {vocab_words}

                Before moving onto the next word, try to make sure the user has passed the drill.  However, give up after 3 or 4 attempts
                if it's too difficult for the user.  If the user is struggling, try to give them a hint or explain the word in their native language.

                Here are some ideas for drill formats you can use, but feel free to come up with your own:

                - Translation drills: "How do you say [native word] in {self.onboarding_data.target_language}?"
                - Definition drills: "What does [target word] mean in {self.onboarding_data.native_language}?"
                - Context drills: "Use the word [target word] in a sentence"
                - Reverse translation: "What's the {self.onboarding_data.native_language} word for [target word]?"
                - Listening comprehension: "Speak out 3-4 sentences in {self.onboarding_data.native_language}, and ask the user questions to test their comprehension of the sentences, focusing on the vocabulary words in the questions."

                After finishing with drills, avoid asking the user what they want to talk about, since they often don't know. Instead, suggest topics based on their interests and previous conversations.
                """

                logfire.info(
                    "Sending function call result for user_wants_vocab_drill",
                    message=message,
                )

                # Send function call result back to the API
                result_message = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": data.get("call_id"),
                        "output": json.dumps(
                            {
                                "success": True,
                                "message": message,
                            }
                        ),
                    },
                }

                if not self.websocket or self.websocket.closed:
                    raise RuntimeError(
                        "WebSocket connection is not open. Cannot send function call result."
                    )

                await self.websocket.send(json.dumps(result_message))

                # Trigger response generation after function call to keep conversation going
                response_message = {"type": "response.create"}
                await self.websocket.send(json.dumps(response_message))

            else:
                raise RuntimeError(f"Unsupported function call: {function_name}. ")

        except Exception as e:
            logfire.exception(f"Error processing function call: {e}")
            return

    async def process_vocab_function_call(self, function_name: str, data):
        if function_name == "user_used_other_language_mistake":
            arguments = json.loads(data.get("arguments", "{}"))
            llm_response = arguments.get("mistake_explanation", "")
            message = await self._user_used_other_language_mistake(llm_response)
        elif function_name == "user_asked_for_translation":
            arguments = json.loads(data.get("arguments", "{}"))
            logfire.info(
                "Processing user_asked_for_translation function call",
                arguments=arguments,
            )
            llm_response = arguments.get("words_needing_translation", "")
            message = await self._user_asked_for_translation(llm_response)
        elif function_name == "user_asked_to_save_vocab_word":
            arguments = json.loads(data.get("arguments", "{}"))
            logfire.info(
                "Processing user_asked_to_save_vocab_word function call",
                arguments=arguments,
            )
            llm_response = arguments.get("word_or_phrase_to_save", "")
            message = await self._user_asked_to_save_vocab_word(llm_response)

        logfire.info(
            f"Sending  function call result: {message}",
        )

        # Send function call result back to the API
        result_message = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": data.get("call_id"),
                "output": json.dumps(
                    {
                        "success": True,
                        "message": message,
                    }
                ),
            },
        }

        if not self.websocket or self.websocket.closed:
            raise RuntimeError(
                "WebSocket connection is not open. Cannot send function call result."
            )

        await self.websocket.send(json.dumps(result_message))

        # Trigger response generation after function call to keep conversation going
        response_message = {"type": "response.create"}
        await self.websocket.send(json.dumps(response_message))

        # Schedule vocabulary extraction after a short delay since otherwise this seems
        # slow down the response generation.  Not sure why
        async def delayed_extract():
            await asyncio.sleep(1)
            await self._extract_vocabulary_from_llm_response(llm_response)

        asyncio.create_task(delayed_extract())

    async def _connect_websocket_start_audio_streams(self):
        if not await self._connect_websocket():
            return False

        # Store event loop for thread communication
        self.loop = asyncio.get_event_loop()

        # Start audio streams
        self.is_recording = True
        self.is_playing = True
        self._start_audio_input_stream()
        self._start_audio_output_stream()

    async def start_conversation(self):
        """Start the realtime audio conversation."""

        await self._connect_websocket_start_audio_streams()

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
