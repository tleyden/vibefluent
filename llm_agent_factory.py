from typing import Type, TypeVar
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from constants import DEFAULT_MODEL, DEFAULT_FALLBACK_MODEL, DEFAULT_REASONING_MODEL

T = TypeVar("T")


class LLMAgentFactory:
    def __init__(self):
        # Setup fallback model: try Claude Sonnet first, then OpenAI GPT
        self.model = FallbackModel(
            AnthropicModel(DEFAULT_MODEL), OpenAIModel(DEFAULT_FALLBACK_MODEL)
        )

        self.reasoning_model = FallbackModel(
            OpenAIModel(DEFAULT_REASONING_MODEL), AnthropicModel(DEFAULT_MODEL)
        )

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