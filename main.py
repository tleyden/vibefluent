from onboarding import load_onboarding_data, run_onboarding


def main():
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

    # TODO: Add main application logic here
    print("VibeFluent main application would start here...")


if __name__ == "__main__":
    main()
