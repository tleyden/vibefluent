from typing import List, Optional
from llm_agent_factory import LLMAgentFactory
from models import OnboardingData
from models import VocabWord
from database import get_database
import random
from pydantic import BaseModel
import logfire


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
        self.drill_conversation_history: List[str] = []  # Track drill session history

        drill_agent_system_prompt = self._generate_system_prompt()
        self.drill_agent_system_prompt = drill_agent_system_prompt
        self.agent = self.factory.create_agent(
            result_type=DrillResponse,
            system_prompt=drill_agent_system_prompt,
        )
        logfire.info(
            f"VocabDrillAgent initialized for {self.onboarding_data.name} ",
            system_prompt=drill_agent_system_prompt,
        )

        # Create evaluation agent
        eval_agent_system_prompt = self._generate_evaluation_prompt()
        self.eval_agent_system_prompt = eval_agent_system_prompt
        self.evaluator = self.factory.create_agent(
            result_type=AnswerEvaluation,
            system_prompt=eval_agent_system_prompt,
        )
        logfire.info(
            f"VocabDrillAgent Evaluator initialized for {self.onboarding_data.name} ",
            system_prompt=eval_agent_system_prompt,
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
        6. Test the user's memory recall abilities, so avoid using both the native language word 
           and target language translation in the same drill, since it doesn't test their recall.
           For example, don't ask |How would you say "The scientist is in the field of quantum physics" in Spanish, using the word "ámbito"?|
           Because it contains both "field" and "ámbito" in the same question.
        
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
        self.current_vocab_words = self.db.get_all_vocab_words(self.onboarding_data)
        self.drill_conversation_history = []  # Reset conversation history for new session

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

        # Create conversation history context
        history_context = ""
        if self.drill_conversation_history:
            recent_history = self.drill_conversation_history[-10:]  # Last 10 exchanges
            history_context = "\n\nRecent drill conversation history:\n" + "\n".join(
                recent_history
            )

        prompt = f"""
        Create a vocabulary drill for this word pair:
        - {self.onboarding_data.target_language}: {current_word.word_in_target_language}
        - {self.onboarding_data.native_language}: {current_word.word_in_native_language}
        
        Choose one of the drill formats randomly and create an engaging question.
        Make sure it's suitable for voice interaction (no visual elements needed).
        
        {history_context}
        
        Consider the conversation history above to:
        1. Vary the drill format from what was recently used
        2. Build on previous context if relevant
        3. Keep the session engaging and progressive
        """

        logfire.info(
            f"Generating drill for {self.onboarding_data.name}",
            prompt=prompt,
            current_word=current_word,
            history_length=len(self.drill_conversation_history),
        )
        result = self.agent.run_sync(prompt)

        logfire.info(
            f"Drill agent result for {self.onboarding_data.name}",
            result=result.data,
            prompt=prompt,
            system_prompt=self.drill_agent_system_prompt,
            current_word=current_word,
            onboarding_data=self.onboarding_data,
        )

        # Add the generated drill to conversation history
        self.drill_conversation_history.append(
            f"VibeFluent: {result.data.drill_question}"
        )

        self.current_word_index += 1
        return result.data

    def evaluate_answer(self, user_answer: str, expected: str) -> str:
        """Evaluate user's answer using LLM agent for more sophisticated judgment."""
        # Add user's answer to conversation history
        self.drill_conversation_history.append(f"User: {user_answer}")

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

        logfire.info(
            f"Evaluating answer for {self.onboarding_data.name}",
            prompt=prompt,
            user_answer=user_answer,
            expected=expected,
        )
        result = self.evaluator.run_sync(prompt)

        logfire.info(
            f"Evaluation agent result for {self.onboarding_data.name}",
            result=result.data,
            prompt=prompt,
            system_prompt=self.eval_agent_system_prompt,
            user_answer=user_answer,
            expected=expected,
            onboarding_data=self.onboarding_data,
        )

        evaluation = result.data

        feedback = ""
        if evaluation.is_correct:
            feedback = f"Excellent! {evaluation.feedback} 🎉"
        else:
            feedback = f"{evaluation.feedback} Let's keep practicing!"

        # Add feedback to conversation history
        self.drill_conversation_history.append(f"VibeFluent: {feedback}")

        return feedback

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
