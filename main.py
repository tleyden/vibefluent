from onboarding import load_onboarding_data, run_onboarding
from conversation import ConversationAgent
from dotenv import load_dotenv


def run_conversation_loop(onboarding_data):
    """Run the main conversation loop."""
    print("\n" + "=" * 60)
    print("🌍 Welcome to your conversation practice! 🌍")
    print("Type 'quit' or 'exit' to end the conversation")
    print("=" * 60 + "\n")

    # Initialize conversation agent
    conversation_agent = ConversationAgent(onboarding_data)

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

            # Skip empty inputs
            if not user_input:
                print("Sorry, I didn't get anything.  Try again!\n")
                continue

            # Get AI response
            print("\nThinking... 🤔")
            response = conversation_agent.get_response(user_input)

            if response.vocab_words_to_practice:
                print(
                    "\nVibeFluent: Here are some vocabulary words to practice based on your message:\n"
                    f"{', '.join(str(word) for word in response.vocab_words_to_practice)}\n"
                )

            print(f"\nVibeFluent: {response.assistant_message}")
            print(f"\nVibeFluent: {response.follow_up_question}\n")

        except KeyboardInterrupt:
            print(f"\n\nGoodbye, {onboarding_data.name}! Thanks for practicing! 👋")
            break
        except Exception as e:
            print(f"\nSorry, I encountered an error: {e}")
            print("Let's keep chatting! What would you like to chat about?\n")


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
