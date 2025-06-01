from onboarding import load_onboarding_data, run_onboarding
from conversation import run_text_conversation_loop
from realtime_audio_conversation import run_realtime_conversation_loop, run_realtime_drill_loop
from constants import MODE, REALTIME_AUDIO_CONVERSATION, REALTIME_AUDIO_DRILL, TEXT
from dotenv import load_dotenv
import logfire
import os
import signal
import sys


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\nGoodbye! Thanks for using VibeFluent! 👋")
    sys.exit(0)


def main():
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
                # console=False,
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
    # Initialize agents and database based on mode
    if MODE == REALTIME_AUDIO_CONVERSATION:
        run_realtime_conversation_loop(onboarding_data)
    elif MODE == REALTIME_AUDIO_DRILL:
        run_realtime_drill_loop(onboarding_data)
    else:  # Default to TEXT mode
        run_text_conversation_loop(onboarding_data)


if __name__ == "__main__":
    main()
