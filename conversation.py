from typing import List
from llm_agent_factory import LLMAgentFactory
from onboarding import OnboardingData
from models import ConversationResponse
from database import get_database

class ConversationAgent:
    def __init__(self, onboarding_data: OnboardingData):
        self.onboarding_data = onboarding_data
        self.conversation_history: List[str] = []
        self.max_history = 100
        self.factory = LLMAgentFactory()
        self.db = get_database()


        # Create the conversation agent
        self.agent = self.factory.create_agent(
            result_type=ConversationResponse,
            system_prompt=self._generate_system_prompt(),
        )

    def _generate_system_prompt(self) -> str:
        return f"""
        You are a friendly, enthusiastic conversation partner helping {self.onboarding_data.name} practice {self.onboarding_data.target_language}.
        
        User Profile:
        - Name: {self.onboarding_data.name}
        - Native Language: {self.onboarding_data.native_language}
        - Learning Target Language: {self.onboarding_data.target_language}
        - Current Level: {self.onboarding_data.target_language_level}
        - Interests: {self.onboarding_data.conversation_interests}
        - Learning Goal: {self.onboarding_data.reason_for_learning}
        - Vocabulary Words To Practice: {self.db.get_all_vocab_words()}
        
        Your role:
        1. Make the conversation feel natural and enjoyable
        2. Adapt your language complexity to their {self.onboarding_data.target_language_level} level
        3. Be encouraging and supportive of their language learning journey
        4. Ask engaging follow-up questions to keep the conversation flowing
        5. Occasionally incorporate their interests: {self.onboarding_data.conversation_interests}
        6. Help them practice by gently correcting errors when appropriate
        7. Keep things fairly brief, because users get overwhelmed with long messages
        8. Try to integrate some of their vocabulary words to practice into the conversation naturally, 
           but avoid doing so if there are already multiple usage instances in the conversation history.
        
        Response format:
        - assistant_message: Your main response to their message (encouraging, helpful)
        - follow_up_question: An engaging question to continue the conversation
        - vocab_words_user_asked_about: Did the user explicitly ask what a word means in their current message?
                                   If so, extract vocabulary words they might want to practice.  Otherwise, leave empty.

        Keep responses conversational, warm, and appropriately challenging for their level.
        """

    def generate_initial_question(self) -> str:
        """Generate a personalized question using the LLM based on user's interests."""

        # Create a simple agent for generating initial questions
        question_agent = self.factory.create_agent(
            result_type=str,
            system_prompt=f"""
            You are helping {self.onboarding_data.name} practice {self.onboarding_data.target_language}.
            Generate an engaging conversation starter question that relates to their interests.
            
            User Profile:
            - Learning: {self.onboarding_data.target_language} (Level: {self.onboarding_data.target_language_level})
            - Interests: {self.onboarding_data.conversation_interests}
            - Learning Goal: {self.onboarding_data.reason_for_learning}
            
            Create a friendly, engaging question that:
            1. Relates to their stated interests: {self.onboarding_data.conversation_interests}
            2. Is appropriate for their {self.onboarding_data.target_language_level} level
            3. Will lead to an interesting conversation
            4. Is open-ended and encourages them to share their thoughts/experiences
            5. Keep things fairly brief, because users get overwhelmed with long messages
            
            Return only the question text, nothing else.
            """,
        )

        prompt = f"""
        Generate a conversation starter question for {self.onboarding_data.name} that relates to their interests: {self.onboarding_data.conversation_interests}.
        
        The question should be engaging, open-ended, and help them practice their {self.onboarding_data.target_language}.
        Make it personal and interesting based on what they've shared about their interests.
        """

        try:
            result = question_agent.run_sync(prompt)
            generated_question = result.data.strip()
            return f"Hello {self.onboarding_data.name}! Let's practice your {self.onboarding_data.target_language}. {generated_question}"
        except Exception:
            # Fallback to a generic question if LLM fails
            return f"Hello {self.onboarding_data.name}! Let's practice your {self.onboarding_data.target_language} by talking about {self.onboarding_data.conversation_interests}. What would you like to share about this topic?"

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
        
        with recent context: {recent_context}
        
        Please respond in a way that:
        1. Acknowledges what they shared 
        2. Keeps the conversation engaging
        3. Helps them practice their {self.onboarding_data.target_language}
        4. Shows genuine interest in their thoughts/experiences
        5. Asks a follow-up question to continue the conversation naturally
        
        
        Additional instructions:

        - vocab_words_user_asked_about: Did the user explicitly ask what a word 
                                        means in their current message |{user_message}|?
                                        For example: "How do you say X" or "What does X mean"?  
                                        If so, extract vocabulary words.
        
        Remember their level is {self.onboarding_data.target_language_level}, so adjust your language complexity accordingly.
        """

        result = self.agent.run_sync(prompt)
        return result.data
