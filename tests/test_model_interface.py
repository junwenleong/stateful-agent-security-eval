"""Property tests for Model_Interface factory and temperature invariant."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings, HealthCheck

from src.agent.model_interface import (
    AnthropicInterface,
    ModelConfig,
    OllamaInterface,
    OpenAIInterface,
    create_model_interface,
)
from tests.strategies import providers

# ---------------------------------------------------------------------------
# Property 1: Model_Interface Factory Correctness
# Validates: Requirements 1.2, 1.3
# ---------------------------------------------------------------------------

_PROVIDER_MODEL = {
    "openai": "gpt-4o-mini-2024-07-18",
    "anthropic": "claude-3-5-haiku-20241022",
    "ollama": "llama3.1:8b",
}

_PROVIDER_CLASS = {
    "openai": OpenAIInterface,
    "anthropic": AnthropicInterface,
    "ollama": OllamaInterface,
}


@given(provider=providers)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_property1_factory_returns_correct_subclass(provider: str) -> None:
    """**Validates: Requirements 1.2, 1.3**

    For any valid provider in {openai, anthropic, ollama},
    create_model_interface(config) returns the correct subclass.
    """
    model_name = _PROVIDER_MODEL[provider]
    expected_class = _PROVIDER_CLASS[provider]

    if provider == "openai":
        env_var = "TEST_OPENAI_KEY"
        config = ModelConfig(provider=provider, model_name=model_name, api_key_env=env_var)
        with patch.dict(os.environ, {env_var: "sk-dummy"}):
            with patch("openai.OpenAI"):
                interface = create_model_interface(config)
    elif provider == "anthropic":
        env_var = "TEST_ANTHROPIC_KEY"
        config = ModelConfig(provider=provider, model_name=model_name, api_key_env=env_var)
        with patch.dict(os.environ, {env_var: "sk-ant-dummy"}):
            with patch("anthropic.Anthropic"):
                interface = create_model_interface(config)
    else:  # ollama — no API key needed
        config = ModelConfig(provider=provider, model_name=model_name)
        interface = create_model_interface(config)

    assert isinstance(interface, expected_class), (
        f"Expected {expected_class.__name__} for provider={provider!r}, "
        f"got {type(interface).__name__}"
    )


# ---------------------------------------------------------------------------
# Property 21: Temperature Invariant
# Validates: Requirements 9.8
# ---------------------------------------------------------------------------

from hypothesis import strategies as st
from src.agent.model_interface import ModelConfig as _ModelConfig


@given(
    provider=providers,
    model_name=st.just("gpt-4o-mini-2024-07-18"),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property21_temperature_default_is_zero(provider: str, model_name: str) -> None:
    """**Validates: Requirements 9.8**

    For any ModelConfig created without specifying temperature,
    temperature field is 0.0 by default.
    """
    name = _PROVIDER_MODEL[provider]
    config = _ModelConfig(provider=provider, model_name=name)
    assert config.temperature == 0.0, (
        f"Expected temperature=0.0 for provider={provider!r}, got {config.temperature}"
    )
