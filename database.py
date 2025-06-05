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
from typing import List, Tuple
from datetime import datetime
import asyncio
import concurrent.futures
from models import VocabWord
from models import OnboardingData

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

    # Relationship to vocab records
    vocab_records = relationship("VocabRecord", back_populates="onboarding_record")


class VocabRecord(Base):
    __tablename__ = "vocab"

    id = Column(Integer, primary_key=True)
    onboarding_record_id = Column(Integer, ForeignKey("onboarding.id"), nullable=False)
    native_language = Column(String, nullable=False)
    target_language = Column(String, nullable=False)
    vocab_word_native = Column(String, nullable=False)
    vocab_word_target = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship to tracking records
    tracking_records = relationship("VocabTracking", back_populates="vocab_record")
    # Relationship to onboarding record
    onboarding_record = relationship("OnboardingRecord", back_populates="vocab_records")


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
            return OnboardingData(
                id=record.id,
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
        # Get the current onboarding record to link vocab words to it
        onboarding_record = self.session.query(OnboardingRecord).first()
        if not onboarding_record:
            logfire.warning(
                "No onboarding record found. Cannot save vocab words without user context."
            )
            return

        for vocab_word in vocab_words:
            # Check if this vocab pair already exists for this user
            existing_record = (
                self.session.query(VocabRecord)
                .filter_by(
                    onboarding_record_id=onboarding_record.id,
                    vocab_word_native=vocab_word.word_in_native_language,
                    vocab_word_target=vocab_word.word_in_target_language,
                )
                .first()
            )

            if not existing_record:
                # Create new record
                record = VocabRecord(
                    onboarding_record_id=onboarding_record.id,
                    native_language=native_language,
                    target_language=target_language,
                    vocab_word_native=vocab_word.word_in_native_language,
                    vocab_word_target=vocab_word.word_in_target_language,
                )
                self.session.add(record)

        self.session.commit()

    def get_all_vocab_words(
        self, onboarding_data: OnboardingData, limit: int = None
    ) -> List[VocabWord]:
        """Get all vocabulary words for the given user."""
        if onboarding_data.id is None:
            logfire.warning("OnboardingData has no ID. Cannot retrieve vocab words.")
            return []

        query = self.session.query(VocabRecord).filter_by(
            onboarding_record_id=onboarding_data.id
        )

        if limit:
            query = query.limit(limit)

        records = query.all()

        # ## experimental testing
        # spaced_repetition_words = self.get_vocab_words_for_spaced_repetition(
        #     onboarding_data=onboarding_data, limit=20
        # )
        # logfire.info(
        #     f"Retrieved {len(records)} vocab words and {len(spaced_repetition_words)} spaced repetition vocab words for user {onboarding_data.name}",
        #     spaced_repetition_words=spaced_repetition_words,
        # )

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
                passed,
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

        # Get the current onboarding record
        onboarding_record = self.session.query(OnboardingRecord).first()
        if not onboarding_record:
            logfire.warning(
                "No onboarding record found. Cannot save drill result without user context."
            )
            return

        # Find the corresponding vocab record for this user
        vocab_record = (
            self.session.query(VocabRecord)
            .filter_by(
                onboarding_record_id=onboarding_record.id,
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
                f"Could not find vocab record for {vocab_word_target_language} for current user. Ignoring drill result."
            )

    def get_vocab_words_for_spaced_repetition(
        self, onboarding_data: OnboardingData, limit: int = 10
    ) -> List[Tuple[VocabWord, int, datetime, int, float]]:
        """Get vocabulary words ordered by spaced repetition priority.

        Returns tuples of (VocabWord, attempts_count, last_attempt_date, correct_attempts, priority).
        """
        if onboarding_data.id is None:
            logfire.warning(
                "OnboardingData has no ID. Cannot retrieve spaced repetition words."
            )
            return []

        from sqlalchemy import case, func, cast, Float
        from sqlalchemy.sql.functions import coalesce
        from datetime import datetime

        # Subquery to get latest tracking info for each vocab word
        latest_tracking = (
            self.session.query(
                VocabTracking.vocab_record_id,
                func.count(VocabTracking.id).label("total_attempts"),
                func.sum(case((VocabTracking.result == True, 1), else_=0)).label(
                    "correct_attempts"
                ),
                func.max(VocabTracking.drill_date).label("last_attempt"),
            )
            .group_by(VocabTracking.vocab_record_id)
            .subquery()
        )

        # Main query joining vocab records with tracking data, filtered by user AND target language
        query = (
            self.session.query(
                VocabRecord,
                coalesce(latest_tracking.c.total_attempts, 0).label("attempts_count"),
                latest_tracking.c.last_attempt.label("last_attempt_date"),
                coalesce(latest_tracking.c.correct_attempts, 0).label(
                    "correct_attempts"
                ),
            )
            .outerjoin(
                latest_tracking, VocabRecord.id == latest_tracking.c.vocab_record_id
            )
            .filter(VocabRecord.onboarding_record_id == onboarding_data.id)
            .filter(VocabRecord.target_language == onboarding_data.target_language)
        )

        # Calculate priority for spaced repetition
        # Priority factors:
        # 1. Never attempted words get highest priority (with recent words prioritized)
        # 2. Words with low success rate get higher priority
        # 3. Words not seen recently get higher priority

        now = datetime.now()

        # Calculate days since creation (for prioritizing recent additions)
        # Use cast to Float to ensure we get decimal precision
        days_since_creation = case(
            (VocabRecord.created_at.is_(None), 999.0),
            else_=cast(
                func.julianday(now) - func.julianday(VocabRecord.created_at), Float
            ),
        )

        # Calculate days since last attempt (0 if never attempted)
        days_since_last = case(
            (latest_tracking.c.last_attempt.is_(None), 0.0),
            else_=cast(
                func.julianday(now) - func.julianday(latest_tracking.c.last_attempt),
                Float,
            ),
        )

        # Calculate success rate (0 if never attempted)
        success_rate = case(
            (latest_tracking.c.total_attempts == 0, 0.0),
            else_=cast(latest_tracking.c.correct_attempts, Float)
            * 1.0
            / cast(latest_tracking.c.total_attempts, Float),
        )

        # Priority score: higher is better
        # For never attempted words: prioritize recent additions
        # Use a larger base number and ensure we maintain decimal precision
        priority_score = case(
            # Never attempted - recent words get higher priority
            # Use 10000 as base and subtract days_since_creation to ensure positive priority
            (latest_tracking.c.total_attempts == 0, 10000.0 - days_since_creation),
            # Low success rate - add recency bonus
            (success_rate < 0.5, 5000.0 + days_since_last),
            # Normal spaced repetition - based on time since last attempt
            else_=days_since_last,
        )

        # Add debugging columns to see what's happening
        query = query.add_columns(
            priority_score.label("priority"),
            days_since_creation.label("debug_days_since_creation"),
            days_since_last.label("debug_days_since_last"),
            success_rate.label("debug_success_rate"),
            latest_tracking.c.total_attempts.label("debug_total_attempts"),
        )

        # Order by priority (highest first)
        query = query.order_by(priority_score.desc()).limit(limit)

        results = []
        for (
            record,
            attempts_count,
            last_attempt_date,
            correct_attempts,
            priority,
            debug_days_since_creation,
            debug_days_since_last,
            debug_success_rate,
            debug_total_attempts,
        ) in query.all():
            vocab_word = VocabWord(
                word_in_target_language=record.vocab_word_target,
                word_in_native_language=record.vocab_word_native,
            )

            # Debug logging for each word
            logfire.info(
                f"Vocab word debug info: {vocab_word.word_in_target_language}",
                word=vocab_word.word_in_target_language,
                created_at=record.created_at,
                attempts_count=attempts_count,
                priority=priority,
                debug_days_since_creation=debug_days_since_creation,
                debug_days_since_last=debug_days_since_last,
                debug_success_rate=debug_success_rate,
                debug_total_attempts=debug_total_attempts,
            )

            results.append(
                (
                    vocab_word,
                    attempts_count,
                    last_attempt_date,
                    correct_attempts,
                    priority,
                )
            )

        logfire.info(
            f"Retrieved {len(results)} vocab words for spaced repetition",
            user_id=onboarding_data.id,
            target_language=onboarding_data.target_language,
            results_count=len(results),
            current_time=now,
        )

        return results

    def _calculate_spaced_repetition_priority(
        self,
        attempts: int,
        correct: int,
        last_attempt: datetime | None,
        now: datetime,
        created_at: datetime,
    ) -> float:
        """Calculate priority score for spaced repetition.

        Higher score = more urgent to review
        Prioritizes recently added words first.
        """
        # Calculate days since word was created
        days_since_creation = (now - created_at).total_seconds() / (24 * 3600)

        # Boost priority for recently added words (within last 7 days)
        recency_boost = 0
        if days_since_creation <= 7:
            recency_boost = (7 - days_since_creation) * 50  # Up to 350 point boost

        # Never attempted - very high priority, especially if recent
        if attempts == 0:
            return 1000.0 + recency_boost

        # Calculate success rate
        success_rate = correct / attempts if attempts > 0 else 0.0

        # Calculate days since last attempt
        if last_attempt:
            days_since = (now - last_attempt).total_seconds() / (24 * 3600)
        else:
            days_since = 999  # Very old or never attempted

        # Spaced repetition intervals based on success rate
        if success_rate >= 0.8:  # Well known - longer intervals
            target_interval = min(30, 2 ** (correct - 1))  # Exponential up to 30 days
        elif success_rate >= 0.6:  # Moderately known
            target_interval = min(7, correct + 1)  # Linear up to 7 days
        else:  # Poorly known - shorter intervals
            target_interval = max(1, correct * 0.5)

        # Priority increases as we exceed the target interval
        if days_since >= target_interval:
            # Overdue - priority increases with how overdue it is
            priority = 100 + (days_since - target_interval) * 10
        else:
            # Not yet due - lower priority
            priority = 10 - (target_interval - days_since)

        # Boost priority for words with low success rates
        if success_rate < 0.5:
            priority *= 2

        # Add recency boost to final priority
        priority += recency_boost

        return max(0, priority)


# Global database instance
_database = None


def get_database() -> Database:
    """Get or create database instance."""
    global _database
    if _database is None:
        _database = Database()
    return _database
