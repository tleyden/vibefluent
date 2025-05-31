from typing import List, Optional
from llm_agent_factory import LLMAgentFactory
from onboarding import OnboardingData
from models import VocabWord
from database import get_database
import random
from pydantic import BaseModel


class DrillResponse(BaseModel):
    drill_question: str
    expected_answer: str


class AnswerEvaluation(BaseModel):
    is_correct: bool
    feedback: str


class VocabDrillAgent:
    def __init__(self, onboarding_data: OnboardingData):
        self.onboarding_data = onboarding_data
        self.factory = LLMAgentFactory()
        self.db = get_database()
        self.current_vocab_words: List[VocabWord] = []
        self.current_word_index = 0
        self.agent = self.factory.create_agent(
            result_type=DrillResponse,
            system_prompt=self._generate_system_prompt(),
        )
        
        # Create evaluation agent
        self.evaluator = self.factory.create_agent(
            result_type=AnswerEvaluation,
            system_prompt=self._generate_evaluation_prompt(),
        )

    def _generate_system_prompt(self) -> str:
        return f"""
        You are a vocabulary drill instructor helping {self.onboarding_data.name} practice {self.onboarding_data.target_language} vocabulary.
        
        User Profile:
        - Name: {self.onboarding_data.name}
        - Native Language: {self.onboarding_data.native_language}
        - Learning: {self.onboarding_data.target_language}
        - Level: {self.onboarding_data.target_language_level}
        
        Your role:
        1. Create engaging vocabulary drills that are hands-free friendly
        2. Design questions that only require spoken answers (no visual elements)
        3. Adapt difficulty to their {self.onboarding_data.target_language_level} level
        4. Be encouraging and supportive
        5. Keep instructions clear and concise
        6. Don't give away the answers directly as part of the question
        
        Drill formats to use:
        - Translation drills: "How do you say [native word] in {self.onboarding_data.target_language}?"
        - Definition drills: "What does [target word] mean in {self.onboarding_data.native_language}?"
        - Context drills: "Use the word [target word] in a sentence"
        - Reverse translation: "What's the {self.onboarding_data.native_language} word for [target word]?"
        
        Response format:
        - drill_question: The question/prompt for the user (voice-friendly)
        - expected_answer: The correct answer you're looking for
        
        Keep everything concise and voice-interaction friendly.
        """
    
    def _generate_evaluation_prompt(self) -> str:
        return f"""
        You are evaluating {self.onboarding_data.name}'s answers to vocabulary drills for learning {self.onboarding_data.target_language}.
        
        Your role:
        1. Judge if their answer is correct or acceptable
        2. Be lenient with minor spelling/pronunciation variations
        3. Accept synonyms and reasonable alternative answers
        4. For sentence creation, check if they used the word correctly in context
        5. Provide encouraging, constructive feedback
        6. Consider their {self.onboarding_data.target_language_level} level when evaluating
        
        Response format:
        - is_correct: True if the answer is correct/acceptable, False otherwise
        - feedback: Encouraging feedback with the correct answer if needed
        
        Be supportive and help them learn!
        """

    def start_drill_session(self) -> str:
        """Start a new drill session with available vocabulary."""
        self.current_vocab_words = self.db.get_all_vocab_words()

        if not self.current_vocab_words:
            return "You don't have any vocabulary words to practice yet! Try having a conversation first and ask about some words."

        # Shuffle for variety
        random.shuffle(self.current_vocab_words)
        self.current_word_index = 0

        return f"Great! Let's practice your {len(self.current_vocab_words)} vocabulary words. Say 'exit drill' to return to conversation mode."

    def get_next_drill(self) -> Optional[DrillResponse]:
        """Get the next vocabulary drill."""
        if not self.current_vocab_words or self.current_word_index >= len(
            self.current_vocab_words
        ):
            return None

        current_word = self.current_vocab_words[self.current_word_index]

        prompt = f"""
        Create a vocabulary drill for this word pair:
        - {self.onboarding_data.target_language}: {current_word.word_in_target_language}
        - {self.onboarding_data.native_language}: {current_word.word_in_native_language}
        
        Choose one of the drill formats randomly and create an engaging question.
        Make sure it's suitable for voice interaction (no visual elements needed).
        """

        try:
            result = self.agent.run_sync(prompt)
            self.current_word_index += 1
            return result.data
        except Exception:
            # Fallback drill
            drill_types = [
                f"How do you say '{current_word.word_in_native_language}' in {self.onboarding_data.target_language}?",
                f"What does '{current_word.word_in_target_language}' mean in {self.onboarding_data.native_language}?",
                f"Use the word '{current_word.word_in_target_language}' in a sentence.",
            ]

            question = random.choice(drill_types)
            expected = (
                current_word.word_in_target_language
                if "How do you say" in question
                else current_word.word_in_native_language
            )

            return DrillResponse(
                drill_question=question,
                expected_answer=expected,
            )

    def evaluate_answer(self, user_answer: str, expected: str) -> str:
        """Evaluate user's answer using LLM agent for more sophisticated judgment."""
        prompt = f"""
        Question context: The user was asked to provide: {expected}
        User's answer: "{user_answer}"
        Expected answer: "{expected}"
        
        Please evaluate if the user's answer is correct or acceptable for a {self.onboarding_data.target_language_level} level learner.
        Consider:
        - Exact matches
        - Close spelling variations
        - Synonyms or alternative valid translations
        - For sentence usage, whether the word is used correctly in context
        - Minor grammatical errors that don't affect meaning
        
        Be encouraging and supportive in your feedback.
        """
        
        try:
            result = self.evaluator.run_sync(prompt)
            evaluation = result.data
            
            if evaluation.is_correct:
                return f"Excellent! {evaluation.feedback} 🎉"
            else:
                return f"{evaluation.feedback} Let's keep practicing!"
                
        except Exception:
            # Fallback to simple comparison if LLM fails
            user_clean = user_answer.strip().lower()
            expected_clean = expected.strip().lower()
            
            if (
                user_clean == expected_clean
                or user_clean in expected_clean
                or expected_clean in user_clean
            ):
                return "Excellent! That's correct! 🎉"
            else:
                return f"Not quite. The answer was: {expected}. Let's keep practicing!"

    def get_session_progress(self) -> str:
        """Get current progress in the drill session."""
        if not self.current_vocab_words:
            return "No active drill session."

        completed = min(self.current_word_index, len(self.current_vocab_words))
        total = len(self.current_vocab_words)

        if completed >= total:
            return (
                f"Drill session complete! You practiced all {total} words. Great job!"
            )
        else:
            return f"Progress: {completed}/{total} words completed."
