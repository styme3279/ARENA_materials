"""
Provider configuration dataclass for LLM API requests.

This module provides a clean configuration interface for managing
LLM generation parameters across different providers.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any


MODEL2PROVIDER = {
    "qwen3-30b-a3b": "fireworks",
    "Qwen/Qwen2.5-14B": "together",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": "together",
    "llama-v3p3-70b-instruct": "fireworks",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": "together",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": "together",
    "deepseek/deepseek-r1-distill-llama-70b": "together",
}


@dataclass
class ProviderConfig:
    """Configuration for LLM provider requests.

    This dataclass encapsulates all common parameters used for text generation
    across different LLM providers (Fireworks, OpenRouter, etc.).

    Attributes:
        temperature: Sampling temperature (0.0 to 2.0)
        top_p: Nucleus sampling parameter (0.0 to 1.0)
        max_tokens: Maximum tokens to generate
        model: Model identifier (e.g., "qwen3-30b-a3b")
        max_retries: Maximum retry attempts for failed requests
        verbose: Whether to print debug information
        logprobs: Number of logprob alternatives (max 5 for most providers)
        echo: Whether to return logprobs for prompt tokens (completions API only)
        top_k: Top-k sampling parameter
        presence_penalty: Penalty for token presence (-2.0 to 2.0)
        frequency_penalty: Penalty for token frequency (-2.0 to 2.0)

    Example:
        # Create with defaults
        config = ProviderConfig()

        # Create with custom values
        config = ProviderConfig(
            temperature=0.9,
            max_tokens=4096,
            model="llama-3-70b"
        )

        # Modify after creation
        config.temperature = 0.5
        config.echo = False

        # Convert to dict for API calls
        params = config.to_dict()
    """

    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 4096
    model: str = "qwen3-30b-a3b"
    max_retries: int = 200
    verbose: bool = False
    logprobs: int = 5
    echo: bool = True
    top_k: int = 40
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    req_exist: bool = False
    provider: str = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for API calls.

        Returns:
            Dictionary of all configuration parameters.
        """
        return asdict(self)

    def to_api_params(self, exclude: Optional[list] = None) -> Dict[str, Any]:
        """Convert to API parameters, excluding specified fields.

        Args:
            exclude: List of field names to exclude (e.g., ['verbose', 'max_retries'])

        Returns:
            Dictionary of API parameters.
        """
        params = self.to_dict()
        if exclude:
            for field_name in exclude:
                params.pop(field_name, None)
        return params

    def copy_with(self, **kwargs) -> "ProviderConfig":
        """Create a copy with modified parameters.

        Args:
            **kwargs: Parameters to override in the copy

        Returns:
            New ProviderConfig instance with modified parameters.

        Example:
            base_config = ProviderConfig(temperature=0.7)
            high_temp_config = base_config.copy_with(temperature=1.2)
        """
        current_values = self.to_dict()
        current_values.update(kwargs)
        return ProviderConfig(**current_values)

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If any parameter is out of valid range.
        """
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {self.temperature}")

        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(f"top_p must be between 0.0 and 1.0, got {self.top_p}")

        if self.max_tokens < 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")

        if self.logprobs < 0 or self.logprobs > 20:
            raise ValueError(f"logprobs must be between 0 and 20, got {self.logprobs}")

        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError(
                f"presence_penalty must be between -2.0 and 2.0, got {self.presence_penalty}"
            )

        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError(
                f"frequency_penalty must be between -2.0 and 2.0, got {self.frequency_penalty}"
            )

        if self.top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {self.top_k}")

    def fix_model_name(self):
        # switch between openrouter and together model name?
        if self.model == r"deepseek/deepseek-r1-distill-llama-70b":
            self.model = r"deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

    def __post_init__(self):
        """Validate parameters after initialization."""
        self.validate()
        # self.fix_model_name()
        if self.provider is None:
            self.provider = MODEL2PROVIDER[self.model]


# if __name__ == "__main__":
# Preset configurations for common use cases
PRESETS = {
    "deterministic": ProviderConfig(
        temperature=0.0,
        top_p=1.0,
        top_k=1,
    ),
    "creative": ProviderConfig(
        temperature=1.2,
        top_p=0.95,
        top_k=50,
        frequency_penalty=0.3,
        presence_penalty=0.3,
    ),
    "balanced": ProviderConfig(
        temperature=0.7,
        top_p=0.95,
        top_k=40,
    ),
    "focused": ProviderConfig(
        temperature=0.3,
        top_p=0.9,
        top_k=20,
    ),
    "exploration": ProviderConfig(
        temperature=1.5,
        top_p=0.98,
        top_k=100,
        frequency_penalty=0.5,
    ),
}


def get_preset(name: str) -> ProviderConfig:
    """Get a preset configuration by name.

    Args:
        name: Preset name (deterministic, creative, balanced, focused, exploration)

    Returns:
        ProviderConfig instance with preset values.

    Raises:
        KeyError: If preset name not found.

    Example:
        config = get_preset("creative")
        config.model = "llama-3-70b"  # Customize after getting preset
    """
    if name not in PRESETS:
        raise KeyError(f"Preset '{name}' not found. Available: {list(PRESETS.keys())}")
    return PRESETS[name].copy_with()
