from sqlalchemy import create_engine, Column, String, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import List
from datetime import datetime
from models import VocabWord
import asyncio
import concurrent.futures


Base = declarative_base()


class OnboardingRecord(Base):
    __tablename__ = "onboarding"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    native_language = Column(String, nullable=False)
    target_language = Column(String, nullable=False)
    conversation_interests = Column(String, nullable=False)
    target_language_level = Column(String, nullable=False)
    reason_for_learning = Column(String, nullable=False)


class VocabRecord(Base):
    __tablename__ = "vocab"

    id = Column(Integer, primary_key=True)
    native_language = Column(String, nullable=False)
    target_language = Column(String, nullable=False)
    vocab_word_native = Column(String, nullable=False)
    vocab_word_target = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Database:
    def __init__(self, db_path: str = "vibefluent.db"):
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def save_onboarding_data(self, onboarding_data) -> None:
        """Save or update onboarding data in SQLite."""
        # Delete existing record (we only store one user's data)
        self.session.query(OnboardingRecord).delete()

        # Create new record
        record = OnboardingRecord(
            name=onboarding_data.name,
            native_language=onboarding_data.native_language,
            target_language=onboarding_data.target_language,
            conversation_interests=onboarding_data.conversation_interests,
            target_language_level=onboarding_data.target_language_level,
            reason_for_learning=onboarding_data.reason_for_learning,
        )

        self.session.add(record)
        self.session.commit()

    def load_onboarding_data(self):
        """Load onboarding data from SQLite."""
        record = self.session.query(OnboardingRecord).first()
        if record:
            from onboarding import OnboardingData

            return OnboardingData(
                name=record.name,
                native_language=record.native_language,
                target_language=record.target_language,
                conversation_interests=record.conversation_interests,
                target_language_level=record.target_language_level,
                reason_for_learning=record.reason_for_learning,
            )
        return None

    def delete_onboarding_data(self) -> None:
        """Delete all onboarding data from SQLite."""
        self.session.query(OnboardingRecord).delete()
        self.session.commit()

    async def save_vocab_words_async(
        self, vocab_words, native_language, target_language
    ):
        """Async version of save_vocab_words that doesn't block the event loop."""
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            await loop.run_in_executor(
                executor,
                self.save_vocab_words,
                vocab_words,
                native_language,
                target_language,
            )

    def save_vocab_words(
        self, vocab_words: List[VocabWord], native_language: str, target_language: str
    ) -> None:
        """Save vocabulary words to the vocab table."""
        for vocab_word in vocab_words:
            # Check if this vocab pair already exists
            existing = (
                self.session.query(VocabRecord)
                .filter_by(
                    native_language=native_language,
                    target_language=target_language,
                    vocab_word_native=vocab_word.word_in_native_language,
                    vocab_word_target=vocab_word.word_in_target_language,
                )
                .first()
            )

            # Only add if it doesn't already exist
            if not existing:
                vocab_record = VocabRecord(
                    native_language=native_language,
                    target_language=target_language,
                    vocab_word_native=vocab_word.word_in_native_language,
                    vocab_word_target=vocab_word.word_in_target_language,
                )
                self.session.add(vocab_record)

        self.session.commit()

    def get_vocab_words(
        self, native_language: str, target_language: str
    ) -> List[VocabRecord]:
        """Get all vocabulary words for a language pair."""
        return (
            self.session.query(VocabRecord)
            .filter_by(
                native_language=native_language,
                target_language=target_language,
            )
            .order_by(VocabRecord.created_at.desc())
            .all()
        )

    def get_all_vocab_words(self, onboarding_data) -> List:
        """Get all vocabulary words as VocabWord instances for the user's language pair."""
        from models import VocabWord

        records = (
            self.session.query(VocabRecord)
            .filter_by(
                native_language=onboarding_data.native_language,
                target_language=onboarding_data.target_language,
            )
            .order_by(VocabRecord.created_at.desc())
            .all()
        )

        vocab_words = []
        for record in records:
            vocab_word = VocabWord(
                word_in_target_language=record.vocab_word_target,
                word_in_native_language=record.vocab_word_native,
            )
            vocab_words.append(vocab_word)

        return vocab_words

    def close(self):
        """Close database session."""
        self.session.close()


# Global database instance
_db_instance = None


def get_database() -> Database:
    """Get or create database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance
