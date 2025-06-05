from typing import List
from constants import MODE
from drill import VocabDrillAgent
from llm_agent_factory import LLMAgentFactory
from models import ConversationResponse, OnboardingData
from database import get_database
from prompt_manager import get_prompt_manager
import logfire
import sys
import readline


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


def run_text_conversation_loop(onboarding_data: OnboardingData):
    """Run the main conversation loop with drill mode support."""
    print("\n" + "=" * 60)
    print("🌍 Welcome to your conversation practice! 🌍")
    print(f"Mode: {MODE}")
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'drill mode' to enter vocabulary drill mode")
    print("Type 're-onboard' to update your profile settings")
    print("Press Ctrl+C to exit anytime")
    print("=" * 60 + "\n")

    conversation_agent = ConversationAgent(onboarding_data)

    drill_agent = VocabDrillAgent(onboarding_data)
    db = get_database()

    # App state
    is_drill_mode = False
    current_drill = None

    # Start with initial question
    initial_question = conversation_agent.generate_initial_question()
    print(f"VibeFluent: {initial_question}\n")

    while True:
        try:
            # Get user input with improved handling
            user_input = get_user_input("You: ")

            # Check for exit commands
            if user_input.lower() in ["quit", "exit", "q"]:
                print(
                    f"\nGoodbye, {onboarding_data.name}! Keep practicing your {onboarding_data.target_language}! 🎉"
                )
                break

            # Check for re-onboard command
            if user_input == "re-onboard":
                print("\n🔄 Re-onboarding Mode! 🔄")
                print("Let's update your profile settings...")

                # Delete existing onboarding data
                db.delete_onboarding_data()

                # Run onboarding again
                from onboarding import run_onboarding

                new_onboarding_data = run_onboarding()

                print(f"\nProfile updated successfully, {new_onboarding_data.name}!")
                print(
                    f"Now ready to help you learn {new_onboarding_data.target_language}!"
                )

                # Reinitialize agents with new data
                conversation_agent = ConversationAgent(new_onboarding_data)
                drill_agent = VocabDrillAgent(new_onboarding_data)
                onboarding_data = new_onboarding_data

                # Reset drill mode state
                is_drill_mode = False
                current_drill = None

                # Generate new initial question
                initial_question = conversation_agent.generate_initial_question()
                print(f"\nVibeFluent: {initial_question}\n")
                continue

            # Check for mode switching commands
            if user_input.lower() in ["drill mode", "start drill", "drill"]:
                is_drill_mode = True
                start_message = drill_agent.start_drill_session()
                print("\n🎯 Entering Drill Mode! 🎯")
                print(f"VibeFluent: {start_message}\n")

                # Get first drill
                current_drill = drill_agent.get_next_drill()
                if current_drill:
                    print(f"VibeFluent: {current_drill.drill_question}")
                continue

            if user_input.lower() in ["exit drill", "conversation mode", "end drill"]:
                if is_drill_mode:
                    is_drill_mode = False
                    current_drill = None
                    # Clear drill conversation history when exiting drill mode
                    drill_agent.drill_conversation_history = []
                    print("\n💬 Returning to Conversation Mode! 💬")
                    print(
                        "VibeFluent: Welcome back! What would you like to chat about?\n"
                    )
                continue

            # Skip empty inputs
            if not user_input:
                mode_text = "drill" if is_drill_mode else "conversation"
                print(
                    f"Sorry, I didn't get anything. Try again! (Currently in {mode_text} mode)\n"
                )
                continue

            # Handle drill mode
            if is_drill_mode:
                if user_input.lower() in ["next", "next drill", "skip"]:
                    current_drill = drill_agent.get_next_drill()
                    if current_drill:
                        print(f"\nVibeFluent: {current_drill.drill_question}")
                    else:
                        print(f"\nVibeFluent: {drill_agent.get_session_progress()}")
                        print(
                            "Say 'drill mode' to start a new session or 'exit drill' to return to conversation.\n"
                        )
                else:
                    # Evaluate answer
                    if current_drill:
                        feedback = drill_agent.evaluate_answer(
                            user_input, current_drill.expected_answer
                        )
                        print(f"\nVibeFluent: {feedback}")

                        # Automatically get next drill
                        current_drill = drill_agent.get_next_drill()
                        if current_drill:
                            print(f"\nVibeFluent: {current_drill.drill_question}")
                        else:
                            print(f"\n{drill_agent.get_session_progress()}")
                            print(
                                "Say 'drill mode' to start a new session or 'exit drill' to return to conversation.\n"
                            )
                    else:
                        print(
                            "\nVibeFluent: No active drill. Say 'drill mode' to start or 'exit drill' to return to conversation.\n"
                        )
                continue

            # Handle conversation mode
            print("\nThinking... 🤔")
            response = conversation_agent.get_response(user_input)

            # Save vocab words to database if any were returned
            if response.vocab_words_user_asked_about:
                db.save_vocab_words(
                    response.vocab_words_user_asked_about,
                    onboarding_data.native_language,
                    onboarding_data.target_language,
                )

                logfire.info(
                    f"New vocabulary words saved {', '.join(str(word) for word in response.vocab_words_user_asked_about)}",
                    onboarding_data=onboarding_data,
                    vocab_words=response.vocab_words_user_asked_about,
                )
                print(
                    f"New vocabulary words saved {', '.join(str(word) for word in response.vocab_words_user_asked_about)}"
                )

            print(f"\nVibeFluent: {response.assistant_message}")
            print(f"\nVibeFluent: {response.follow_up_question}\n")

        except KeyboardInterrupt:
            print(f"\n\nGoodbye, {onboarding_data.name}! Thanks for practicing! 👋")
            sys.exit(0)
        except Exception as e:
            print(f"\nSorry, I encountered an error: {e}")
            mode_text = "drill" if is_drill_mode else "conversation"
            print(f"Let's keep going! (Currently in {mode_text} mode)\n")


def get_user_input(prompt: str = "You: ") -> str:
    """Get user input with better handling for different environments."""
    try:
        # Flush output to ensure prompt appears
        print(prompt, end="", flush=True)
        readline.get_history_length()  # Just make sure readline isn't removed from imports
        user_input = input().strip()
        return user_input
    except (EOFError, KeyboardInterrupt):
        # Handle Ctrl+C and Ctrl+D
        print("\n\nGoodbye! Thanks for using VibeFluent! 👋")
        sys.exit(0)
