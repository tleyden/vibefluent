from pydantic import BaseModel
from typing import List, Optional
from enum import Enum


class VocabWord(BaseModel):
    word_in_target_language: str
    word_in_native_language: str

    def __str__(self):
        return f"{self.word_in_target_language} ({self.word_in_native_language})"

class Mode(Enum):
    CONVERSATION_MODE = "conversation_mode"
    DRILL_MODE = "drill_mode"

class ConversationResponse(BaseModel):
    assistant_message: str
    follow_up_question: str
    vocab_words_user_asked_about: List[VocabWord] = []
    user_requested_mode_change: Optional[Mode] = None
