from pydantic import BaseModel
from typing import List


class VocabWord(BaseModel):
    word_in_target_language: str
    word_in_native_language: str

    def __str__(self):
        return f"{self.word_in_target_language} ({self.word_in_native_language})"


class ConversationResponse(BaseModel):
    assistant_message: str
    follow_up_question: str
    vocab_words_user_asked_about: List[VocabWord] = []



class VocabExtractResponse(BaseModel):
    vocab_words: List[VocabWord] = []
