from pydantic import BaseModel
from typing import List
from llm_agent_factory import LLMAgentFactory
from onboarding import OnboardingData
import random


class ConversationResponse(BaseModel):
    assistant_message: str
    follow_up_question: str


class TravelConversationAgent:
    def __init__(self, onboarding_data: OnboardingData):
        self.onboarding_data = onboarding_data
        self.conversation_history: List[str] = []
        self.max_history = 100
        self.factory = LLMAgentFactory()

        # Create the conversation agent
        self.agent = self.factory.create_agent(
            result_type=ConversationResponse,
            system_prompt=self._generate_system_prompt(),
        )

    def _generate_system_prompt(self) -> str:
        return f"""
        You are a friendly, enthusiastic travel conversation partner helping {self.onboarding_data.name} practice {self.onboarding_data.target_language}.
        
        User Profile:
        - Name: {self.onboarding_data.name}
        - Native Language: {self.onboarding_data.native_language}
        - Learning: {self.onboarding_data.target_language}
        - Current Level: {self.onboarding_data.target_language_level}
        - Interests: {self.onboarding_data.conversation_interests}
        - Learning Goal: {self.onboarding_data.reason_for_learning}
        
        Your role:
        1. Keep conversations focused on travel-related topics
        2. Adapt your language complexity to their {self.onboarding_data.target_language_level} level
        3. Be encouraging and supportive of their language learning journey
        4. Ask engaging follow-up questions to keep the conversation flowing
        5. Occasionally incorporate their interests: {self.onboarding_data.conversation_interests}
        6. Help them practice by gently correcting errors when appropriate
        7. Make the conversation feel natural and enjoyable
        
        Response format:
        - assistant_message: Your main response to their message (encouraging, helpful, travel-focused)
        - follow_up_question: An engaging question to continue the conversation
        
        Keep responses conversational, warm, and appropriately challenging for their level.
        """

    def generate_initial_question(self) -> str:
        """Generate a random travel-related question to start the conversation."""
        travel_questions = [
            "What's the most interesting place you've ever visited or would like to visit?",
            "If you could travel anywhere in the world right now, where would you go and why?",
            "What's your favorite type of accommodation when traveling - hotels, hostels, or something else?",
            "Do you prefer planning every detail of a trip or being spontaneous? Tell me about your travel style!",
            "What's the best local food you've tried while traveling, or what would you most like to try?",
            "Would you rather explore a bustling city or relax in nature during your travels?",
            "What's one travel experience that's on your bucket list?",
            "Do you prefer traveling solo, with friends, or with family? What do you enjoy about that style?",
            "What's the longest trip you've ever taken or would like to take?",
            "If you could learn about any culture through travel, which would fascinate you most?",
        ]

        question = random.choice(travel_questions)
        return f"Hello {self.onboarding_data.name}! Let's practice your {self.onboarding_data.target_language} by chatting about travel. {question}"

    def add_to_history(self, user_message: str):
        """Add user message to conversation history, maintaining max limit."""
        self.conversation_history.append(user_message)
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history :]

    def get_response(self, user_message: str) -> ConversationResponse:
        """Get AI response to user message, incorporating conversation history."""
        self.add_to_history(user_message)

        # Create context from recent conversation history
        recent_context = ""
        if len(self.conversation_history) > 1:
            recent_messages = self.conversation_history[
                -10:
            ]  # Last 10 messages for context
            recent_context = "\n\nRecent conversation context:\n" + "\n".join(
                f"User: {msg}" for msg in recent_messages
            )

        prompt = f"""
        The user just said: "{user_message}"
        
        {recent_context}
        
        Please respond in a way that:
        1. Acknowledges what they shared about travel
        2. Keeps the conversation engaging and travel-focused
        3. Helps them practice their {self.onboarding_data.target_language}
        4. Shows genuine interest in their travel thoughts/experiences
        5. Asks a follow-up question to continue the conversation naturally
        
        Remember their level is {self.onboarding_data.target_language_level}, so adjust your language complexity accordingly.
        """

        result = self.agent.run_sync(prompt)
        return result.data
