from typing import List
from llm_agent_factory import LLMAgentFactory
from onboarding import OnboardingData
from models import ConversationResponse
from database import get_database
from prompt_manager import get_prompt_manager
import logfire


class ConversationAgent:
    def __init__(self, onboarding_data: OnboardingData):
        self.onboarding_data = onboarding_data
        self.conversation_history: List[str] = []
        self.max_history = 100
        self.factory = LLMAgentFactory()
        self.db = get_database()
        self.prompt_manager = get_prompt_manager()

        # Create the conversation agent using template
        vocab_words = self.db.get_all_vocab_words(self.onboarding_data)
        system_prompt = self.prompt_manager.render_conversation_system_prompt(
            self.onboarding_data, vocab_words, mode="text"
        )
        self.conversation_system_prompt = system_prompt
        self.agent = self.factory.create_agent(
            result_type=ConversationResponse,
            system_prompt=system_prompt,
        )
        logfire.info(
            f"ConversationAgent initialized for {self.onboarding_data.name} ",
            system_prompt=system_prompt,
        )

    def generate_initial_question(self) -> str:
        """Generate a personalized question using the LLM based on user's interests."""
        # Create a simple agent for generating initial questions using template
        system_prompt = self.prompt_manager.render_initial_question_prompt(
            self.onboarding_data
        )
        self.initial_question_system_prompt = system_prompt
        question_agent = self.factory.create_agent(
            result_type=str,
            system_prompt=system_prompt,
        )

        prompt = f"""
        Generate a conversation starter question for {self.onboarding_data.name} that relates to their interests: {self.onboarding_data.conversation_interests}.
        
        The question should be engaging, open-ended, and help them practice their {self.onboarding_data.target_language}.
        Make it personal and interesting based on what they've shared about their interests.
        """

        result = question_agent.run_sync(prompt)

        logfire.info(
            f"Initial question agent result for {self.onboarding_data.name}",
            result=result.data,
            prompt=prompt,
            system_prompt=self.initial_question_system_prompt,
            onboarding_data=self.onboarding_data,
        )

        generated_question = result.data.strip()
        return f"Hello {self.onboarding_data.name}! Let's practice your {self.onboarding_data.target_language}. {generated_question}"

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

        logfire.info(
            f"Generating question for {self.onboarding_data.name}",
            prompt=prompt,
            onboarding_data=self.onboarding_data,
        )
        result = self.agent.run_sync(prompt)

        logfire.info(
            f"Conversation agent result for {self.onboarding_data.name}",
            result=result.data,
            prompt=prompt,
            system_prompt=self.conversation_system_prompt,
            onboarding_data=self.onboarding_data,
        )

        return result.data
