"""
LLM Provider Abstraction Layer

Supports multiple LLM backends:
- Ollama (local or remote)
- vLLM  (OpenAI-compatible server – ideal for HPC / SLURM jobs)
- HuggingFace Transformers (direct GPU inference)
- Mock (testing)
"""

from abc import ABC, abstractmethod
from typing import Dict, List
import logging
from config import OLLAMA_BASE_URL, VLLM_BASE_URL, DEBUG_MODE

try:
    import ollama
except ImportError:  # Optional dependency when only HF/mock providers are used
    ollama = None

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response from LLM."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM provider is available."""
        pass


class OllamaProvider(LLMProvider):
    """Ollama LLM provider for local/remote model serving."""

    def __init__(self, model_name: str, base_url: str = OLLAMA_BASE_URL, timeout: int = 300):
        """
        Initialize Ollama provider.

        Args:
            model_name: Name of the Ollama model (e.g., "qwen3:8b")
            base_url: Ollama API URL (default: http://localhost:11434)
            timeout: Request timeout in seconds
        """
        if ollama is None:
            raise ImportError(
                "Ollama Python package is not installed. "
                "Install with `pip install ollama` or switch to `--provider hf`."
            )

        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout
        self.client = ollama.Client(host=base_url)

        if DEBUG_MODE:
            logger.info(f"Initialized OllamaProvider: {model_name} at {base_url}")

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate response using Ollama.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters (temperature, top_p, etc.)

        Returns:
            Generated response text
        """
        try:
            # Newer ollama-python versions expect sampling params under `options`
            # rather than as top-level kwargs in chat().
            option_keys = {
                "temperature", "top_p", "top_k", "seed", "num_predict",
                "repeat_penalty", "repeat_last_n", "mirostat", "mirostat_eta",
                "mirostat_tau", "tfs_z", "num_ctx", "num_gpu", "num_thread",
                "stop"
            }

            options = dict(kwargs.pop("options", {}) or {})
            for key in list(kwargs.keys()):
                if key in option_keys:
                    options[key] = kwargs.pop(key)

            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                **kwargs,
            }
            if options:
                payload["options"] = options

            response = self.client.chat(**payload)
            return response.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Error generating response from Ollama: {e}")
            raise

    def is_available(self) -> bool:
        """Check if Ollama model is available."""
        try:
            # Try to pull model info to verify availability
            response = self.client.show(self.model_name)
            return response is not None
        except Exception as e:
            logger.warning(f"Ollama model '{self.model_name}' not available: {e}")
            return False


class MockProvider(LLMProvider):
    """Mock provider for testing without LLM (returns placeholder responses)."""

    def __init__(self, model_name: str = "mock"):
        self.model_name = model_name

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Return a mock response."""
        return f"[MOCK RESPONSE from {self.model_name}] This is a test response."

    def is_available(self) -> bool:
        """Mock provider is always available."""
        return True


class VLLMProvider(LLMProvider):
    """
    vLLM provider – talks to a running vLLM OpenAI-compatible server.

    Typical usage on HPC (inside a SLURM job):
      1. Start vLLM server:
           python -m vllm.entrypoints.openai.api_server \\
               --model Qwen/Qwen3-14B-Instruct \\
               --served-model-name qwen3-14b \\
               --port 8000 --dtype float16
      2. Point this provider at http://localhost:8000/v1
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = VLLM_BASE_URL,
        max_tokens: int = 1024,
        timeout: int = 300,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "VLLMProvider requires the `openai` package. "
                "Install with: pip install openai"
            ) from exc

        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens

        # vLLM does not need a real API key – use a dummy value
        self._client = OpenAI(
            api_key="vllm-no-key",
            base_url=f"{self.base_url}/v1",
            timeout=timeout,
        )

        if DEBUG_MODE:
            logger.info(f"Initialized VLLMProvider: model={model_name} at {base_url}")

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate a response using the vLLM OpenAI-compatible endpoint.

        Args:
            messages: List of {role, content} dicts.
            **kwargs: temperature, top_p, max_tokens, etc.

        Returns:
            Generated response text.
        """
        temperature = float(kwargs.pop("temperature", 0.2))
        top_p = float(kwargs.pop("top_p", 0.9))
        max_tokens = int(kwargs.pop("max_tokens", self.max_tokens))

        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                **kwargs,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Error generating response from vLLM: {e}")
            raise

    def is_available(self) -> bool:
        """Ping the vLLM /health endpoint."""
        try:
            import urllib.request
            url = f"{self.base_url}/health"
            with urllib.request.urlopen(url, timeout=5) as resp:
                return resp.status == 200
        except Exception as e:
            logger.warning(f"vLLM server not reachable at {self.base_url}: {e}")
            return False


class HFTransformersProvider(LLMProvider):
    """Hugging Face Transformers provider for direct GPU/CPU inference."""

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 512,
        trust_remote_code: bool = True,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Transformers provider requires `transformers`, `accelerate`, and `torch`. "
                "Install with `pip install transformers accelerate`."
            ) from exc

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype="auto",
            device_map="auto",
        )

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if DEBUG_MODE:
            logger.info(f"Initialized HFTransformersProvider: {model_name}")

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Fallback if tokenizer does not support chat templates
        lines = []
        for m in messages:
            role = m.get("role", "user").upper()
            content = m.get("content", "")
            lines.append(f"{role}: {content}")
        lines.append("ASSISTANT:")
        return "\n\n".join(lines)

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        temperature = float(kwargs.pop("temperature", 0.2))
        top_p = float(kwargs.pop("top_p", 0.9))
        max_new_tokens = int(kwargs.pop("max_new_tokens", self.max_new_tokens))

        prompt_text = self._build_prompt(messages)
        model_device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        if temperature > 0:
            generation_kwargs.update({
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            })
        else:
            generation_kwargs["do_sample"] = False

        with self._torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def is_available(self) -> bool:
        return self.model is not None and self.tokenizer is not None


class LLMProviderFactory:
    """Factory for creating LLM provider instances."""

    _providers = {
        "ollama": OllamaProvider,
        "vllm":   VLLMProvider,
        "hf":     HFTransformersProvider,
        "mock":   MockProvider,
    }

    @classmethod
    def register_provider(cls, provider_type: str, provider_class: type) -> None:
        """Register a new provider type."""
        cls._providers[provider_type.lower()] = provider_class

    @classmethod
    def create(
        cls,
        provider_type: str = "ollama",
        model_name: str = "qwen3:8b",
        **kwargs
    ) -> LLMProvider:
        """
        Create an LLM provider.

        Args:
            provider_type: Type of provider ("ollama", "mock", etc.)
            model_name: Name of the model
            **kwargs: Provider-specific kwargs

        Returns:
            LLMProvider instance

        Raises:
            ValueError: If provider type is not supported
        """
        provider_class = cls._providers.get(provider_type.lower())
        if not provider_class:
            raise ValueError(
                f"Unknown provider type: {provider_type}. "
                f"Supported: {list(cls._providers.keys())}"
            )

        return provider_class(model_name, **kwargs)


# Convenience function for backward compatibility
def get_llm_provider(
    model_name: str = None,
    provider_type: str = "ollama",
    base_url: str = None,
) -> LLMProvider:
    """
    Get an LLM provider with sensible defaults.

    Args:
        model_name:    Name of the model.
        provider_type: "ollama" | "vllm" | "hf" | "mock"
        base_url:      Base URL for remote providers.
                       Defaults to OLLAMA_BASE_URL for ollama,
                       VLLM_BASE_URL for vllm.

    Returns:
        LLMProvider instance
    """
    if provider_type == "ollama":
        return LLMProviderFactory.create(
            provider_type=provider_type,
            model_name=model_name,
            base_url=base_url or OLLAMA_BASE_URL,
        )
    elif provider_type == "vllm":
        return LLMProviderFactory.create(
            provider_type=provider_type,
            model_name=model_name,
            base_url=base_url or VLLM_BASE_URL,
        )
    else:
        return LLMProviderFactory.create(
            provider_type=provider_type,
            model_name=model_name,
        )


