from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderConfig:
    """Configuration for LLM providers.

    Supported providers:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Normalize provider name and map common aliases."""
    val = (value or "").strip().lower()
    alias_map = {
        "anthorpic": "anthropic",
        "claude": "anthropic",
        "google": "gemini",
        "google-genai": "gemini",
        "openai-compatible": "custom",
        "vllm": "custom",
        "local": "ollama",
    }
    return alias_map.get(val, val)


def build_chat_model(config: ProviderConfig) -> Any:
    """Instantiate a LangChain chat model for the selected provider.

    Imports are resolved lazily to allow running without all provider packages installed.
    """
    provider = normalize_provider(config.provider)

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key or os.getenv("OPENAI_API_KEY"),
            )
        except ImportError as exc:
            raise ImportError("Please install `langchain-openai` to use the OpenAI provider.") from exc

    elif provider == "custom":
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key or os.getenv("CUSTOM_API_KEY", "EMPTY"),
                base_url=config.base_url or os.getenv("CUSTOM_BASE_URL", "http://localhost:8000/v1"),
            )
        except ImportError as exc:
            raise ImportError("Please install `langchain-openai` for custom OpenAI-compatible endpoints.") from exc

    elif provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=config.model_name,
                temperature=config.temperature,
                google_api_key=config.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            )
        except ImportError as exc:
            raise ImportError("Please install `langchain-google-genai` to use Google Gemini.") from exc

    elif provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key or os.getenv("ANTHROPIC_API_KEY"),
            )
        except ImportError as exc:
            raise ImportError("Please install `langchain-anthropic` to use Anthropic Claude.") from exc

    elif provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=config.model_name,
                temperature=config.temperature,
                base_url=config.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            )
        except ImportError as exc:
            raise ImportError("Please install `langchain-ollama` to use Ollama.") from exc

    elif provider == "openrouter":
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key or os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            )
        except ImportError as exc:
            raise ImportError("Please install `langchain-openai` to use OpenRouter.") from exc

    else:
        raise ValueError(f"Unsupported provider: '{config.provider}'. Normalized: '{provider}'")
