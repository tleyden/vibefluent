from typing import List
from onboarding import OnboardingData
from models import ConversationResponse
import logfire


class RealtimeAudioConversationAgent:
    def __init__(self, onboarding_data: OnboardingData):
        self.onboarding_data = onboarding_data
        self.conversation_history: List[str] = []
        self.max_history = 100
        
        logfire.info(
            f"RealtimeAudioConversationAgent initialized for {self.onboarding_data.name} (STUB)",
            onboarding_data=onboarding_data,
        )

    def generate_initial_question(self) -> str:
        """Generate a personalized question for realtime audio mode."""
        # TODO: Implement realtime audio initial question generation
        return f"Hello {self.onboarding_data.name}! Realtime audio mode is not yet implemented. Please switch to TEXT mode."

    def add_to_history(self, user_message: str):
        """Add user message to conversation history, maintaining max limit."""
        # TODO: Implement for realtime audio
        self.conversation_history.append(user_message)
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history :]

    def get_response(self, user_message: str) -> ConversationResponse:
        """Get AI response to user message in realtime audio mode."""
        # TODO: Implement realtime audio conversation handling
        self.add_to_history(user_message)
        
        return ConversationResponse(
            assistant_message="Realtime audio mode is not yet implemented.",
            follow_up_question="Please switch to TEXT mode to continue your conversation.",
            vocab_words_user_asked_about=[]
        )
