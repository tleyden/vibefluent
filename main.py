from onboarding import load_onboarding_data, run_onboarding
from conversation import run_text_conversation_loop
from realtime_audio_conversation import (
    run_realtime_conversation_loop,
)
from constants import MODE, REALTIME_AUDIO_CONVERSATION, TEXT
from dotenv import load_dotenv
import logfire
from logfire import ConsoleOptions
import os
import signal
import sys
import argparse


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\nGoodbye! Thanks for using VibeFluent! 👋")
    sys.exit(0)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="VibeFluent - Language Learning Assistant"
    )
    parser.add_argument(
        "--new-user",
        action="store_true",
        help="Force onboarding of a new user regardless of existing users",
    )
    args = parser.parse_args()

    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    load_dotenv()

    # Initialize logfire
    LOGFIRE_API_KEY = os.environ.get("LOGFIRE_API_KEY", "")
    if LOGFIRE_API_KEY:
        service_name = "vibefluent"
        environment = "development"
        if MODE == TEXT:
            logfire.configure(
                token=LOGFIRE_API_KEY,
                service_name=service_name,
                environment=environment,
                console=False,
            )
        else:
            logfire.configure(
                token=LOGFIRE_API_KEY,
                service_name=service_name,
                environment=environment,
                console=ConsoleOptions(min_log_level="info"),
                # console=False,
            )

        logfire.info("VibeFluent application started successfully")

    # Check if user has completed onboarding
    if args.new_user:
        print("Creating new user profile...")
        onboarding_data = run_onboarding()
        print(f"\nWelcome, {onboarding_data.name}! Onboarding complete.")
        print(f"Ready to help you learn {onboarding_data.target_language}!")
    else:
        onboarding_data = load_onboarding_data()
        if onboarding_data is None:
            print("No profile found. Let's set up your profile first!")
            onboarding_data = run_onboarding()
            print(f"\nWelcome {onboarding_data.name}! Your profile has been saved.")
            print(f"Ready to help you learn {onboarding_data.target_language}!")
        else:
            print(f"Welcome back, {onboarding_data.name}!")
            print(
                f"Continuing your {onboarding_data.target_language} learning journey..."
            )

    # Reload to get the ID assigned by the database
    onboarding_data = load_onboarding_data()

    logfire.info(
        "Onboarding data loaded",
        onboarding_data=onboarding_data,
    )

    # Start the conversation loop
    # Initialize agents and database based on mode
    if MODE == REALTIME_AUDIO_CONVERSATION:
        run_realtime_conversation_loop(onboarding_data)
    else:  # Default to TEXT mode
        run_text_conversation_loop(onboarding_data)


if __name__ == "__main__":
    main()
