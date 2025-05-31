import asyncio
import json
import base64
import websockets
import pyaudio
import wave
import threading
import queue
import os
from typing import List, Optional
from onboarding import OnboardingData
from models import ConversationResponse, VocabWord
from database import get_database
import logfire


class RealtimeAudioConversationAgent:
    def __init__(self, onboarding_data: OnboardingData):
        self.onboarding_data = onboarding_data
        self.conversation_history: List[str] = []
        self.max_history = 100
        self.db = get_database()
        
        # Audio settings
        self.sample_rate = 24000
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        
        # WebSocket and audio state
        self.websocket = None
        self.audio = pyaudio.PyAudio()
        self.is_recording = False
        self.is_playing = False
        self.audio_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # OpenAI API key
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for realtime audio mode")
        
        logfire.info(
            f"RealtimeAudioConversationAgent initialized for {self.onboarding_data.name}",
            onboarding_data=onboarding_data,
        )

    def generate_initial_question(self) -> str:
        """Generate a personalized question for realtime audio mode."""
        return f"Hello {self.onboarding_data.name}! Welcome to realtime audio mode. I'll speak with you to help practice your {self.onboarding_data.target_language}. Press Enter to start recording, and I'll respond with audio. Say 'exit realtime' to return to text mode."

    def _create_system_message(self) -> str:
        """Create the system message for the OpenAI Realtime API."""
        vocab_words = self.db.get_all_vocab_words(self.onboarding_data)
        vocab_context = ""
        if vocab_words:
            vocab_list = [f"{w.word_in_target_language} ({w.word_in_native_language})" for w in vocab_words[:20]]
            vocab_context = f"\nVocabulary words to practice: {', '.join(vocab_list)}"
        
        return f"""
You are a friendly, enthusiastic conversation partner helping {self.onboarding_data.name} practice {self.onboarding_data.target_language} through voice conversation.

User Profile:
- Name: {self.onboarding_data.name}
- Native Language: {self.onboarding_data.native_language}
- Learning Target Language: {self.onboarding_data.target_language}
- Current Level: {self.onboarding_data.target_language_level}
- Interests: {self.onboarding_data.conversation_interests}
- Learning Goal: {self.onboarding_data.reason_for_learning}{vocab_context}

Your role:
1. Speak naturally and conversationally in audio responses
2. Adapt your language complexity to their {self.onboarding_data.target_language_level} level
3. Be encouraging and supportive of their language learning journey
4. Ask engaging follow-up questions to keep the conversation flowing
5. Occasionally incorporate their interests: {self.onboarding_data.conversation_interests}
6. Help them practice by gently correcting errors when appropriate
7. Keep responses conversational length (1-3 sentences typically)
8. Naturally incorporate vocabulary words they're learning
9. Respond with enthusiasm and warmth in your voice

This is a voice conversation, so speak naturally as you would in person. Be encouraging and make learning fun!
"""

    async def _connect_websocket(self):
        """Connect to OpenAI Realtime API via WebSocket."""
        url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        
        try:
            self.websocket = await websockets.connect(url, extra_headers=headers)
            logfire.info(f"Connected to OpenAI Realtime API for {self.onboarding_data.name}")
            
            # Send session configuration
            await self._configure_session()
            return True
        except Exception as e:
            logfire.error(f"Failed to connect to OpenAI Realtime API: {e}")
            return False

    async def _configure_session(self):
        """Configure the OpenAI Realtime session."""
        config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": self._create_system_message(),
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 200
                },
                "tools": [],
                "tool_choice": "auto",
                "temperature": 0.8,
                "max_response_output_tokens": 4096
            }
        }
        
        await self.websocket.send(json.dumps(config))
        logfire.info("Session configured for realtime audio")

    def _start_audio_input_stream(self):
        """Start recording audio from microphone."""
        def audio_input_thread():
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            while self.is_recording:
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    if self.websocket and not self.websocket.closed:
                        asyncio.run_coroutine_threadsafe(
                            self._send_audio_chunk(data),
                            self.loop
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
            message = {
                "type": "input_audio_buffer.append",
                "audio": audio_b64
            }
            await self.websocket.send(json.dumps(message))

    def _start_audio_output_stream(self):
        """Start playing audio from the response queue."""
        def audio_output_thread():
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size
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

    async def _process_websocket_message(self, data):
        """Process different types of messages from the API."""
        message_type = data.get("type")
        
        if message_type == "response.audio.delta":
            # Received audio chunk
            audio_b64 = data.get("delta")
            if audio_b64:
                audio_data = base64.b64decode(audio_b64)
                self.audio_queue.put(audio_data)
                
        elif message_type == "response.audio_transcript.delta":
            # Received transcript chunk
            transcript = data.get("delta", "")
            if transcript:
                # Store partial transcript (could display this in real-time)
                pass
                
        elif message_type == "response.audio_transcript.done":
            # Complete transcript received
            transcript = data.get("transcript", "")
            if transcript:
                self.conversation_history.append(f"Assistant: {transcript}")
                logfire.info(f"Assistant transcript: {transcript}")
                
        elif message_type == "input_audio_buffer.speech_started":
            logfire.info("User started speaking")
            
        elif message_type == "input_audio_buffer.speech_stopped":
            logfire.info("User stopped speaking")
            
        elif message_type == "conversation.item.input_audio_transcription.completed":
            # User's speech was transcribed
            transcript = data.get("transcript", "")
            if transcript:
                self.conversation_history.append(f"User: {transcript}")
                logfire.info(f"User transcript: {transcript}")
                
        elif message_type == "error":
            error_msg = data.get("error", {}).get("message", "Unknown error")
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
            self.conversation_history = self.conversation_history[-self.max_history:]

    def get_response(self, user_message: str) -> ConversationResponse:
        """Get AI response to user message in realtime audio mode."""
        # For compatibility with text mode interface
        # In actual realtime mode, this won't be used much
        self.add_to_history(user_message)
        
        return ConversationResponse(
            assistant_message="In realtime audio mode - responses are provided via voice.",
            follow_up_question="Continue speaking to practice your conversation skills!",
            vocab_words_user_asked_about=[]
        )

    async def send_text_message(self, message: str):
        """Send a text message to the conversation (useful for commands)."""
        if self.websocket and not self.websocket.closed:
            text_message = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": message
                        }
                    ]
                }
            }
            await self.websocket.send(json.dumps(text_message))
            
            # Trigger response generation
            response_message = {
                "type": "response.create"
            }
            await self.websocket.send(json.dumps(response_message))

    def __del__(self):
        """Cleanup audio resources."""
        if hasattr(self, 'audio'):
            self.audio.terminate()
