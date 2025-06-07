from typing import Type, TypeVar
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from constants import DEFAULT_MODEL, DEFAULT_FALLBACK_MODEL, DEFAULT_REASONING_MODEL
import os

T = TypeVar("T")


class LLMAgentFactory:
    def __init__(self):
        # Check if Anthropic API key is available
        has_anthropic_key = bool(os.getenv("ANTHROPIC_API_KEY"))

        if has_anthropic_key:
            # Setup fallback model: try Claude Sonnet first, then OpenAI GPT
            self.model = FallbackModel(
                AnthropicModel(DEFAULT_FALLBACK_MODEL), OpenAIModel(DEFAULT_MODEL)
            )

            self.reasoning_model = FallbackModel(
                OpenAIModel(DEFAULT_REASONING_MODEL),
                AnthropicModel(DEFAULT_FALLBACK_MODEL),
            )
        else:
            # Use only OpenAI models if no Anthropic key
            self.model = OpenAIModel(DEFAULT_MODEL)
            self.reasoning_model = OpenAIModel(DEFAULT_REASONING_MODEL)

    def create_agent(
        self, result_type: Type[T], system_prompt: str = ""
    ) -> Agent[None, T]:
        return Agent(
            model=self.model, system_prompt=system_prompt, result_type=result_type
        )

    def create_reasoning_agent(
        self, result_type: Type[T], system_prompt: str = ""
    ) -> Agent[None, T]:
        return Agent(
            model=self.reasoning_model,
            system_prompt=system_prompt,
            result_type=result_type,
        )
