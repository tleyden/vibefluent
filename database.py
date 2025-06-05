from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    DateTime,
    Boolean,
    ForeignKey,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from typing import List
from datetime import datetime
import asyncio
import concurrent.futures
from models import VocabWord
import logfire

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

    # Relationship to tracking records
    tracking_records = relationship("VocabTracking", back_populates="vocab_record")


class VocabTracking(Base):
    __tablename__ = "vocab_tracking"

    id = Column(Integer, primary_key=True)
    vocab_record_id = Column(Integer, ForeignKey("vocab.id"), nullable=False)
    drill_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    result = Column(Boolean, nullable=False)  # True for pass, False for fail

    # Relationship back to vocab record
    vocab_record = relationship("VocabRecord", back_populates="tracking_records")


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
            existing_record = (
                self.session.query(VocabRecord)
                .filter_by(
                    native_language=native_language,
                    target_language=target_language,
                    vocab_word_native=vocab_word.word_in_native_language,
                    vocab_word_target=vocab_word.word_in_target_language,
                )
                .first()
            )

            if not existing_record:
                # Create new record
                record = VocabRecord(
                    native_language=native_language,
                    target_language=target_language,
                    vocab_word_native=vocab_word.word_in_native_language,
                    vocab_word_target=vocab_word.word_in_target_language,
                )
                self.session.add(record)

        self.session.commit()

    def get_all_vocab_words(
        self, onboarding_data, limit: int = None
    ) -> List[VocabWord]:
        """Get all vocabulary words for the given language pair."""
        query = self.session.query(VocabRecord).filter_by(
            native_language=onboarding_data.native_language,
            target_language=onboarding_data.target_language,
        )

        if limit:
            query = query.limit(limit)

        records = query.all()

        return [
            VocabWord(
                word_in_target_language=record.vocab_word_target,
                word_in_native_language=record.vocab_word_native,
            )
            for record in records
        ]

    async def save_vocab_drill_result_async(
        self,
        vocab_word_target_language: str,
        native_language: str,
        target_language: str,
        passed: bool,
    ):
        """Async version of save_vocab_words that doesn't block the event loop."""
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            await loop.run_in_executor(
                executor,
                self.save_vocab_drill_result,
                vocab_word_target_language,
                native_language,
                target_language,
                passed
            )

    def save_vocab_drill_result(
        self,
        vocab_word_target_language: str,
        native_language: str,
        target_language: str,
        passed: bool,
    ) -> None:
        """Save the result of a vocabulary drill."""

        logfire.info(
            f"Attempting to save drill result for {vocab_word_target_language} - Did it pass: {passed}"
        )

        # Find the corresponding vocab record
        vocab_record = (
            self.session.query(VocabRecord)
            .filter_by(
                native_language=native_language,
                target_language=target_language,
                vocab_word_target=vocab_word_target_language,
            )
            .first()
        )

        if vocab_record:
            logfire.info(
                f"Saving drill result for {vocab_word_target_language} - Did it pass: {passed}"
            )
            # Create tracking record
            tracking_record = VocabTracking(
                vocab_record_id=vocab_record.id,
                result=passed,
            )
            self.session.add(tracking_record)
            self.session.commit()
        else:
            logfire.warning(
                f"Could not find vocab record for {vocab_word_target_language}.  Ignoring drill result."
            )


# Global database instance
_database = None


def get_database() -> Database:
    """Get or create database instance."""
    global _database
    if _database is None:
        _database = Database()
    return _database
