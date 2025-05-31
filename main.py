from onboarding import load_onboarding_data, run_onboarding
from conversation import ConversationAgent
from realtime_audio_conversation import RealtimeAudioConversationAgent
from drill import VocabDrillAgent
from database import get_database
from constants import MODE
from dotenv import load_dotenv
import logfire
import os
import readline
import signal
import sys
import asyncio


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\nGoodbye! Thanks for using VibeFluent! 👋")
    sys.exit(0)


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


def run_conversation_loop(onboarding_data):
    """Run the main conversation loop with drill mode support."""
    print("\n" + "=" * 60)
    print("🌍 Welcome to your conversation practice! 🌍")
    print(f"Mode: {MODE}")
    print("Type 'quit' or 'exit' to end the conversation")
    if MODE != "REALTIME_AUDIO":
        print("Type 'drill mode' to enter vocabulary drill mode")
    print("Type 're-onboard' to update your profile settings")
    print("Press Ctrl+C to exit anytime")
    print("=" * 60 + "\n")

    # Initialize agents and database based on mode
    if MODE == "REALTIME_AUDIO":
        conversation_agent = RealtimeAudioConversationAgent(onboarding_data)
        return run_realtime_audio_loop(conversation_agent, onboarding_data)
    else:  # Default to TEXT mode
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
                if MODE == "REALTIME_AUDIO":
                    conversation_agent = RealtimeAudioConversationAgent(
                        new_onboarding_data
                    )
                else:  # Default to TEXT mode
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

            print(f"\nVibeFluent: {response.assistant_message}")
            print(f"\nVibeFluent: {response.follow_up_question}\n")

        except KeyboardInterrupt:
            print(f"\n\nGoodbye, {onboarding_data.name}! Thanks for practicing! 👋")
            sys.exit(0)
        except Exception as e:
            print(f"\nSorry, I encountered an error: {e}")
            mode_text = "drill" if is_drill_mode else "conversation"
            print(f"Let's keep going! (Currently in {mode_text} mode)\n")


def run_realtime_audio_loop(conversation_agent, onboarding_data):
    """Run the realtime audio conversation loop."""

    async def audio_loop():
        print(f"VibeFluent: {conversation_agent.generate_initial_question()}\n")

        # Start the realtime conversation
        success = await conversation_agent.start_conversation()
        if not success:
            print(
                "Failed to start realtime audio mode. Please check your internet connection and API key."
            )
            return

        print(
            "🎤 Realtime audio mode active! Speak naturally - I'll respond with voice."
        )
        print("Commands you can type:")
        print("- 'quit' or 'exit': End the session")
        print("- 're-onboard': Update your profile")
        print("- Press Enter: Start/stop recording")
        print("=" * 60 + "\n")

        # Create tasks for handling WebSocket and user input
        input_task = asyncio.create_task(
            handle_realtime_input(conversation_agent, onboarding_data)
        )

        try:
            await input_task
        except KeyboardInterrupt:
            print(f"\n\nGoodbye, {onboarding_data.name}! Thanks for practicing! 👋")
        finally:
            await conversation_agent.stop_conversation()

    # Run the async audio loop
    try:
        asyncio.run(audio_loop())
    except Exception as e:
        print(f"Error in realtime audio mode: {e}")
        print("Falling back to text mode...")


async def handle_realtime_input(conversation_agent, onboarding_data):
    """Handle user input during realtime audio mode."""
    loop = asyncio.get_event_loop()

    while True:
        try:
            # Get user input in a non-blocking way
            user_input = await loop.run_in_executor(
                None, get_user_input, "Command (or Enter to toggle recording): "
            )

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

                # Stop current conversation
                await conversation_agent.stop_conversation()

                # Delete existing onboarding data
                from database import get_database

                db = get_database()
                db.delete_onboarding_data()

                # Run onboarding again
                from onboarding import run_onboarding

                new_onboarding_data = run_onboarding()

                print(f"\nProfile updated successfully, {new_onboarding_data.name}!")
                print(
                    f"Now ready to help you learn {new_onboarding_data.target_language}!"
                )

                # Create new conversation agent
                conversation_agent = RealtimeAudioConversationAgent(new_onboarding_data)

                # Restart conversation
                success = await conversation_agent.start_conversation()
                if not success:
                    print("Failed to restart realtime audio mode.")
                    break

                continue

            # Toggle recording on Enter (empty input)
            if not user_input:
                if conversation_agent.is_recording:
                    print("🔴 Recording paused - press Enter to resume")
                    conversation_agent.is_recording = False
                else:
                    print("🎤 Recording resumed - speak naturally!")
                    conversation_agent.is_recording = True
                continue

            # Send text commands to the conversation
            await conversation_agent.send_text_message(user_input)

        except (EOFError, KeyboardInterrupt):
            break


def main():
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    load_dotenv()

    # Initialize logfire
    LOGFIRE_API_KEY = os.environ.get("LOGFIRE_API_KEY", "")
    if LOGFIRE_API_KEY:
        service_name = "vibefluent"
        environment = "development"
        if MODE == "REALTIME_AUDIO":
            logfire.configure(
                token=LOGFIRE_API_KEY,
                service_name=service_name,
                environment=environment,
            )
        else:
            logfire.configure(
                token=LOGFIRE_API_KEY,
                service_name=service_name,
                environment=environment,
                console=False,
            )
            logfire.info("VibeFluent application started successfully")

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
