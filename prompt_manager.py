from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from typing import Dict, Any
from models import OnboardingData
from models import VocabWord
from typing import List


class PromptManager:
    def __init__(self):
        # Set up Jinja2 environment to load templates from the templates directory
        template_dir = Path(__file__).parent / "templates"
        self.env = Environment(loader=FileSystemLoader(template_dir))

    def _get_base_context(self, onboarding_data: OnboardingData) -> Dict[str, Any]:
        """Get base context variables from onboarding data."""
        return {
            "name": onboarding_data.name,
            "native_language": onboarding_data.native_language,
            "target_language": onboarding_data.target_language,
            "target_language_level": onboarding_data.target_language_level,
            "conversation_interests": onboarding_data.conversation_interests,
            "reason_for_learning": onboarding_data.reason_for_learning,
        }

    def render_conversation_system_prompt(
        self,
        onboarding_data: OnboardingData,
        vocab_words: List[VocabWord] = None,
        mode: str = "text",
    ) -> str:
        """Render the conversation system prompt template."""
        template = self.env.get_template("conversation_system_prompt.j2")
        context = self._get_base_context(onboarding_data)
        context.update(
            {"mode": mode, "vocab_words": vocab_words if vocab_words else None}
        )
        return template.render(**context)

    def render_initial_question_prompt(self, onboarding_data: OnboardingData) -> str:
        """Render the initial question generation prompt template."""
        template = self.env.get_template("initial_question_prompt.j2")
        context = self._get_base_context(onboarding_data)
        return template.render(**context)

    def render_vocab_extraction_prompt(self, onboarding_data: OnboardingData) -> str:
        """Render the vocabulary extraction prompt template."""
        template = self.env.get_template("vocab_extraction_prompt.j2")
        context = self._get_base_context(onboarding_data)
        return template.render(**context)

    def render_realtime_session_config(
        self, onboarding_data: OnboardingData, vocab_context: str = ""
    ) -> str:
        """Render the realtime session configuration prompt template."""
        template = self.env.get_template("realtime_session_config.j2")
        context = self._get_base_context(onboarding_data)
        context.update({"vocab_context": vocab_context})
        return template.render(**context)

    async def render_realtime_vocab_extraction_prompt(
        self,
        llm_response: str,
        onboarding_data: OnboardingData,
    ) -> str:
        """Render the realtime vocabulary extraction prompt template."""
        template = self.env.get_template("realtime_vocab_extraction_prompt.j2")
        context = self._get_base_context(onboarding_data)
        context.update(
            {
                "llm_response": llm_response,
            }
        )
        return template.render(**context)


# Global instance
_prompt_manager = None


def get_prompt_manager() -> PromptManager:
    """Get or create prompt manager instance."""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager
