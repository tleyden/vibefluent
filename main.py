from onboarding import load_onboarding_data, run_onboarding
from conversation import ConversationAgent
from drill import VocabDrillAgent
from database import get_database
from dotenv import load_dotenv


def run_conversation_loop(onboarding_data):
    """Run the main conversation loop with drill mode support."""
    print("\n" + "=" * 60)
    print("🌍 Welcome to your conversation practice! 🌍")
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'drill mode' to enter vocabulary drill mode")
    print("=" * 60 + "\n")

    # Initialize agents and database
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
            # Get user input
            user_input = input("You: ").strip()

            # Check for exit commands
            if user_input.lower() in ["quit", "exit", "q"]:
                print(
                    f"\nGoodbye, {onboarding_data.name}! Keep practicing your {onboarding_data.target_language}! 🎉"
                )
                break

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
                    print(f"Hint: {current_drill.encouragement}\n")
                continue

            if user_input.lower() in ["exit drill", "conversation mode", "end drill"]:
                if is_drill_mode:
                    is_drill_mode = False
                    current_drill = None
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
                        print(f"Hint: {current_drill.encouragement}\n")
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
                            print(f"Hint: {current_drill.encouragement}\n")
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

                print(
                    "\nVibeFluent: Here are some vocabulary words to practice based on your message:\n"
                    f"{', '.join(str(word) for word in response.vocab_words_user_asked_about)}\n"
                )

            print(f"\nVibeFluent: {response.assistant_message}")
            print(f"\nVibeFluent: {response.follow_up_question}\n")

        except KeyboardInterrupt:
            print(f"\n\nGoodbye, {onboarding_data.name}! Thanks for practicing! 👋")
            break
        except Exception as e:
            print(f"\nSorry, I encountered an error: {e}")
            mode_text = "drill" if is_drill_mode else "conversation"
            print(f"Let's keep going! (Currently in {mode_text} mode)\n")


def main():
    load_dotenv()

    # Check if user has completed onboarding
    onboarding_data = load_onboarding_data()

    if onboarding_data is None:
        print("Welcome to VibeFluent! Let's get you set up...")
        onboarding_data = run_onboarding()
        print(f"\nWelcome, {onboarding_data.name}! Onboarding complete.")
        print(f"Ready to help you learn {onboarding_data.target_language}!")
    else:
        print(f"Welcome back, {onboarding_data.name}!")
        print(f"Continuing your {onboarding_data.target_language} learning journey...")

    # Start the conversation loop
    run_conversation_loop(onboarding_data)


if __name__ == "__main__":
    main()
