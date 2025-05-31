from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

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
