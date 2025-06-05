import curses
from typing import Optional
from database import get_database
from models import OnboardingData


class OnboardingUI:
    def __init__(self):
        self.questions = [
            ("Name", "What's your name?"),
            ("Native Language", "What's your native language?"),
            ("Target Language", "What language do you want to learn?"),
            ("Conversation Interests", "What topics interest you for conversation?"),
            (
                "Target Language Level",
                "What's your current level? (Beginner/Intermediate/Advanced)",
            ),
            ("Reason for Learning", "Why do you want to learn this language?"),
        ]
        self.answers = {}
        self.current_question = 0

    def run(self, stdscr):
        curses.curs_set(1)  # Show cursor
        stdscr.clear()

        while self.current_question < len(self.questions):
            self.draw_screen(stdscr)
            key = stdscr.getch()

            if key == ord("\n"):  # Enter key
                self.next_question()
            elif key == curses.KEY_BACKSPACE or key == 127:
                self.backspace()
            elif 32 <= key <= 126:  # Printable characters
                self.add_char(chr(key))

        return self.create_onboarding_data()

    def draw_screen(self, stdscr):
        stdscr.clear()
        height, width = stdscr.getmaxyx()

        # Title
        title = "🌟 VibeFluent Onboarding 🌟"
        stdscr.addstr(2, (width - len(title)) // 2, title, curses.A_BOLD)

        # Progress indicator
        progress = f"Question {self.current_question + 1} of {len(self.questions)}"
        stdscr.addstr(4, (width - len(progress)) // 2, progress)

        # Current question
        field_name, question = self.questions[self.current_question]
        stdscr.addstr(7, 4, question, curses.A_BOLD)

        # Current answer
        current_answer = self.answers.get(field_name, "")
        stdscr.addstr(9, 4, f"> {current_answer}")

        # Instructions
        stdscr.addstr(height - 3, 4, "Press ENTER to continue, BACKSPACE to delete")

        # Show previous answers
        y_pos = 12
        for i in range(self.current_question):
            prev_field, prev_question = self.questions[i]
            prev_answer = self.answers.get(prev_field, "")
            if y_pos < height - 5:
                stdscr.addstr(
                    y_pos,
                    4,
                    f"{prev_field}: {prev_answer[:50]}{'...' if len(prev_answer) > 50 else ''}",
                )
                y_pos += 1

        stdscr.refresh()

    def add_char(self, char):
        field_name, _ = self.questions[self.current_question]
        current = self.answers.get(field_name, "")
        self.answers[field_name] = current + char

    def backspace(self):
        field_name, _ = self.questions[self.current_question]
        current = self.answers.get(field_name, "")
        if current:
            self.answers[field_name] = current[:-1]

    def next_question(self):
        field_name, _ = self.questions[self.current_question]
        if self.answers.get(field_name, "").strip():
            self.current_question += 1

    def create_onboarding_data(self) -> OnboardingData:
        return OnboardingData(
            id=None,  # Will be set when saved to database
            name=self.answers.get("Name", ""),
            native_language=self.answers.get("Native Language", ""),
            target_language=self.answers.get("Target Language", ""),
            conversation_interests=self.answers.get("Conversation Interests", ""),
            target_language_level=self.answers.get("Target Language Level", ""),
            reason_for_learning=self.answers.get("Reason for Learning", ""),
        )


def load_onboarding_data() -> Optional[OnboardingData]:
    """Load onboarding data from SQLite database."""
    db = get_database()
    records = db.get_all_onboarding_records()

    if not records:
        return None
    elif len(records) == 1:
        return records[0]
    else:
        # Multiple records - ask user to choose
        print("\nMultiple profiles found:")
        for i, record in enumerate(records, 1):
            print(
                f"{i}. {record.name} - {record.target_language} ({record.target_language_level})"
            )

        while True:
            try:
                choice = (
                    input(f"\nSelect a profile (1-{len(records)}) or 'q' to quit: ")
                    .strip()
                    .lower()
                )
                if choice == "q":
                    return None
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(records):
                    return records[choice_idx]
                else:
                    print(f"Please enter a number between 1 and {len(records)}")
            except (ValueError, KeyboardInterrupt):
                if choice == "q":
                    return None
                print(
                    f"Please enter a number between 1 and {len(records)} or 'q' to quit"
                )


def save_onboarding_data(data: OnboardingData) -> None:
    """Save onboarding data to SQLite database."""
    db = get_database()
    db.save_onboarding_data(data)


def run_onboarding() -> OnboardingData:
    """Run the onboarding process using curses."""
    ui = OnboardingUI()
    data = curses.wrapper(ui.run)
    save_onboarding_data(data)
    return data
