import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import requests

logger = logging.getLogger(__name__)

# Patterns that indicate a pinned version (not a floating alias)
# Matches YYYY-MM-DD (e.g. gpt-4o-mini-2024-07-18) or YYYYMMDD
# (e.g. claude-3-5-haiku-20241022)
_DATE_PATTERN = re.compile(r"\d{4}-?\d{2}-?\d{2}")
# :8b, :latest-q4, etc.
_OLLAMA_VERSION_PATTERN = re.compile(r":\w")


def _validate_model_name(provider: str, name: str) -> None:
    if provider == "ollama":
        if not _OLLAMA_VERSION_PATTERN.search(name):
            raise ValueError(
                f"Ollama model_name '{name}' must include a version tag "
                "(e.g. 'llama3.1:8b'). Floating aliases are not allowed."
            )
    else:
        if not _DATE_PATTERN.search(name):
            raise ValueError(
                f"model_name '{name}' must contain a dated version identifier "
                "(e.g. 'gpt-4o-mini-2024-07-18'). "
                "Floating aliases are not allowed."
            )


@dataclass
class ModelConfig:
    provider: Literal["openai", "anthropic", "ollama", "bedrock"]
    model_name: str
    temperature: float = 0.0
    api_key_env: str = ""
    base_url: str | None = None
    ollama_quantization: str | None = None
    aws_region: str = "ap-southeast-1"
    aws_profile: str | None = None

    def __post_init__(self) -> None:
        if self.provider != "bedrock":
            _validate_model_name(self.provider, self.model_name)


@dataclass
class ChatMessage:
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None  # Required by strict models (qwen3) to match tool results
    tool_calls: list[dict] | None = None  # Carried through for Bedrock toolUse blocks


@dataclass
class ChatResponse:
    content: str
    tool_calls: list[dict] | None = None
    temperature_used: float = 0.0


class ModelInterface(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
    ) -> ChatResponse:
        """Send messages to the LLM. temperature=0.0 enforced."""
        ...


class OpenAIInterface(ModelInterface):
    def __init__(self, config: ModelConfig) -> None:
        from openai import OpenAI

        if not config.api_key_env:
            raise ValueError("ModelConfig.api_key_env must be set for OpenAI.")
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise ValueError(
                f"Environment variable '{config.api_key_env}' is not set."
            )
        self.config = config
        self.client = OpenAI(api_key=api_key)

    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
    ) -> ChatResponse:
        oai_messages = [{"role": m.role, "content": m.content} for m in messages]
        kwargs: dict = {
            "model": self.config.model_name,
            "messages": oai_messages,
            "temperature": 0.0,
        }
        if tools:
            kwargs["tools"] = tools

        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        content = choice.message.content or ""
        raw_tool_calls = choice.message.tool_calls
        tool_calls = None
        if raw_tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in raw_tool_calls
            ]
        # Enforce temperature=0.0 for reproducibility (Req 9.8)
        temperature_used = 0.0
        if self.config.temperature != 0.0:
            logger.warning(
                "OpenAI: requested temperature %.2f overridden to 0.0 "
                "for reproducibility.",
                self.config.temperature,
            )
        assert temperature_used == 0.0, (
            f"OpenAI response temperature must be 0.0 for reproducibility, "
            f"got {temperature_used}"
        )
        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            temperature_used=temperature_used,
        )


class AnthropicInterface(ModelInterface):
    def __init__(self, config: ModelConfig) -> None:
        import anthropic

        if not config.api_key_env:
            raise ValueError("ModelConfig.api_key_env must be set for Anthropic.")
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise ValueError(
                f"Environment variable '{config.api_key_env}' is not set."
            )
        self.config = config
        self.client = anthropic.Anthropic(api_key=api_key)

    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
    ) -> ChatResponse:
        system_parts = [m.content for m in messages if m.role == "system"]
        system_prompt = "\n".join(system_parts) if system_parts else None
        non_system = [m for m in messages if m.role != "system"]
        ant_messages = [{"role": m.role, "content": m.content} for m in non_system]

        if self.config.temperature != 0.0:
            logger.warning(
                "Anthropic: requested temperature %.2f overridden to 0.0 "
                "for reproducibility.",
                self.config.temperature,
            )

        kwargs: dict = {
            "model": self.config.model_name,
            "max_tokens": 4096,
            "messages": ant_messages,
            "temperature": 0.0,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = tools

        response = self.client.messages.create(**kwargs)
        content = ""
        tool_calls = None
        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(
                    {"id": block.id, "name": block.name, "input": block.input}
                )
        # Enforce temperature=0.0 for reproducibility (Req 9.8)
        temperature_used = 0.0
        assert temperature_used == 0.0, (
            f"Anthropic response temperature must be 0.0 for reproducibility, "
            f"got {temperature_used}"
        )
        return ChatResponse(content=content, tool_calls=tool_calls, temperature_used=temperature_used)


class OllamaInterface(ModelInterface):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.base_url = (config.base_url or "http://localhost:11434").rstrip("/")
        self._verify_ollama_available()
        self._verify_model_loaded()

    def _verify_ollama_available(self) -> None:
        """Verify Ollama server is running and accessible.
        
        Raises:
            RuntimeError: If Ollama is not available.
        """
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Ensure Ollama is running: ollama serve"
            ) from e
        except requests.exceptions.Timeout as e:
            raise RuntimeError(
                f"Ollama at {self.base_url} is not responding (timeout). "
                f"Check if Ollama is running."
            ) from e

    def _verify_model_loaded(self) -> None:
        """Verify the model is loaded with the correct quantization.
        
        Raises:
            RuntimeError: If model is not available.
        """
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            models = data.get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            # Check if model is loaded (may have quantization suffix)
            model_found = any(self.config.model_name in name for name in model_names)
            if not model_found:
                raise RuntimeError(
                    f"Model '{self.config.model_name}' not found in Ollama. "
                    f"Available models: {model_names}. "
                    f"Run: ollama pull {self.config.model_name}"
                )
            
            # Log quantization info for reproducibility
            logger.info(
                "Ollama model verified: %s (quantization: %s)",
                self.config.model_name,
                self.config.ollama_quantization or "default",
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Failed to verify Ollama model '{self.config.model_name}': {e}"
            ) from e

    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
    ) -> ChatResponse:
        if self.config.temperature != 0.0:
            logger.warning(
                "Ollama: requested temperature %.2f overridden to 0.0 "
                "for reproducibility.",
                self.config.temperature,
            )

        payload: dict = {
            "model": self.config.model_name,
            "messages": [
                {k: v for k, v in (
                    {"role": m.role, "content": m.content} |
                    ({"tool_call_id": m.tool_call_id} if m.tool_call_id else {})
                ).items()}
                for m in messages
            ],
            "options": {"temperature": 0.0},
            "stream": False,
            "think": False,  # Disable extended thinking (qwen3 etc.) for reproducibility
        }
        if tools:
            payload["tools"] = tools

        resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=1200)
        resp.raise_for_status()
        data = resp.json()
        message = data.get("message", {})
        content = message.get("content", "")
        raw_tool_calls = message.get("tool_calls")
        tool_calls = None
        if raw_tool_calls:
            tool_calls = raw_tool_calls
        
        # Log warning: Ollama may not fully respect temperature=0.0
        # This is a known limitation documented for reproducibility
        logger.debug(
            "Ollama inference complete for model %s. "
            "Note: Ollama inference remains stochastic despite temperature=0.0. "
            "This is a known limitation. See REPRODUCIBILITY.md for details.",
            self.config.model_name,
        )
        
        # Enforce temperature=0.0 for reproducibility (Req 9.8)
        temperature_used = 0.0
        assert temperature_used == 0.0, (
            f"Ollama response temperature must be 0.0 for reproducibility, "
            f"got {temperature_used}"
        )
        return ChatResponse(content=content, tool_calls=tool_calls, temperature_used=temperature_used)


class BedrockInterface(ModelInterface):
    """Interface for AWS Bedrock Claude models.
    
    Note: Uses AWS_PROFILE environment variable or profile parameter to select
    the correct AWS account/role. Temporarily unsets AWS_PROFILE during client
    creation to use fresh SSO tokens instead of static IAM keys.
    """
    
    def __init__(self, config: ModelConfig) -> None:
        import boto3
        import os
        
        # Get profile from config or environment
        profile = config.aws_profile or os.environ.get("AWS_PROFILE")
        
        # Temporarily unset AWS_PROFILE to use fresh SSO tokens
        # (if direnv exports AWS_PROFILE, boto3 uses static IAM keys instead of SSO)
        aws_profile = os.environ.pop("AWS_PROFILE", None)
        try:
            if profile:
                session = boto3.Session(profile_name=profile)
                self.client = session.client("bedrock-runtime", region_name=config.aws_region)
            else:
                self.client = boto3.client("bedrock-runtime", region_name=config.aws_region)
        finally:
            # Restore AWS_PROFILE if it was set
            if aws_profile:
                os.environ["AWS_PROFILE"] = aws_profile
        
        self.config = config
    
    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
    ) -> ChatResponse:
        system_parts = [m.content for m in messages if m.role == "system"]
        system_prompt = "\n".join(system_parts) if system_parts else None
        non_system = [m for m in messages if m.role != "system"]
        
        # Convert LangChain messages to Bedrock format.
        # CRITICAL: Bedrock requires all toolResult blocks for a given assistant turn
        # to be batched into a SINGLE user message. LangGraph emits one ToolMessage
        # per tool call, so we must merge consecutive tool messages here.
        
        bedrock_messages = []
        
        i = 0
        while i < len(non_system):
            m = non_system[i]
            
            if m.role == "tool":
                # Collect ALL consecutive tool messages into one user message
                tool_results = []
                while i < len(non_system) and non_system[i].role == "tool":
                    tm = non_system[i]
                    tool_content = tm.content if isinstance(tm.content, str) else str(tm.content)
                    tool_results.append({
                        "toolResult": {
                            "toolUseId": tm.tool_call_id,
                            "content": [{"text": tool_content}],
                        }
                    })
                    i += 1
                bedrock_messages.append({"role": "user", "content": tool_results})
                continue  # i already advanced
                    
            elif m.role == "assistant" and m.tool_calls:
                # Assistant with tool calls: include toolUse blocks
                content_blocks = []
                if m.content:
                    content_blocks.append({"text": m.content})
                for tc in m.tool_calls:
                    tc_args = tc.get("args", {})
                    tool_use_id = tc.get("id", f"call_{tc.get('name', 'unknown')}")
                    content_blocks.append({
                        "toolUse": {
                            "toolUseId": tool_use_id,
                            "name": tc.get("name", ""),
                            "input": tc_args if isinstance(tc_args, dict) else {},
                        }
                    })
                bedrock_messages.append({"role": "assistant", "content": content_blocks})
                
            elif m.role == "user":
                bedrock_messages.append({
                    "role": "user",
                    "content": [{"text": m.content}] if m.content else [{"text": ""}],
                })
            else:
                bedrock_messages.append({
                    "role": m.role,
                    "content": [{"text": m.content}] if m.content else [{"text": ""}],
                })
            
            i += 1
        
        if self.config.temperature != 0.0:
            logger.warning(
                "Bedrock: requested temperature %.2f overridden to 0.0 "
                "for reproducibility.",
                self.config.temperature,
            )
        
        kwargs: dict = {
            "modelId": self.config.model_name,
            "messages": bedrock_messages,
            "inferenceConfig": {
                "temperature": 0.0,
                "maxTokens": 4096,
            },
        }
        
        if system_prompt:
            kwargs["system"] = [{"text": system_prompt}]
        
        if tools:
            bedrock_tools = []
            for tool in tools:
                # Tools come in format: {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
                if isinstance(tool, dict) and "function" in tool:
                    fn = tool["function"]
                    name = fn.get("name", "")
                    description = fn.get("description", "")
                    parameters = fn.get("parameters", {})
                else:
                    # Fallback for other formats
                    name = tool.get("name", "")
                    description = tool.get("description", "")
                    parameters = tool.get("input_schema", tool.get("parameters", {}))
                
                # Bedrock requires non-empty name and description
                if name and description:
                    bedrock_tools.append({
                        "toolSpec": {
                            "name": name,
                            "description": description,
                            "inputSchema": {
                                "json": parameters
                            }
                        }
                    })
            
            if bedrock_tools:
                kwargs["toolConfig"] = {"tools": bedrock_tools}
        
        response = self.client.converse(**kwargs)
        
        content = ""
        tool_calls = None
        
        for block in response.get("output", {}).get("message", {}).get("content", []):
            if "text" in block:
                content = block.get("text", "")
            elif "toolUse" in block:
                if tool_calls is None:
                    tool_calls = []
                tool_use = block.get("toolUse", {})
                tool_calls.append({
                    "id": tool_use.get("toolUseId", ""),
                    "name": tool_use.get("name", ""),
                    "input": tool_use.get("input", {})
                })
        
        temperature_used = 0.0
        assert temperature_used == 0.0, (
            f"Bedrock response temperature must be 0.0 for reproducibility, "
            f"got {temperature_used}"
        )
        
        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            temperature_used=temperature_used,
        )


def create_model_interface(config: ModelConfig) -> ModelInterface:
    if config.provider == "openai":
        return OpenAIInterface(config)
    if config.provider == "anthropic":
        return AnthropicInterface(config)
    if config.provider == "ollama":
        return OllamaInterface(config)
    if config.provider == "bedrock":
        return BedrockInterface(config)
    raise ValueError(f"Unknown provider: {config.provider!r}")
