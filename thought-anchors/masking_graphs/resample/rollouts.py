"""
Dataclasses for structured LLM response data.

This module provides type-safe dataclass representations of the response
data from LLM APIs, particularly for use with logprobs and caching.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Union


@dataclass
class Usage:
    """Token usage statistics for a response."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class Logprobs:
    """Logprobs data for tokens in a response."""

    tokens: List[str]
    token_logprobs: List[float]
    top_logprobs: List[Dict[str, float]]
    text_offset: List[int]
    prompt_length: int
    num_prompt_tokens: int


@dataclass
class FwResponse:
    """Individual response from the LLM."""

    text: str  # Generated text only (without prompt)
    full_text: str  # Full text including prompt if echo=True
    post: str  # Alias for text (compatibility)
    reasoning: str  # Reasoning content if available
    finish_reason: str  # Why generation stopped (length, stop, etc.)
    provider: str  # Which provider served this
    response_id: str  # Unique response ID
    model: str  # Model that generated this
    object: str  # Object type (e.g., "text_completion")
    created: int  # Unix timestamp
    usage: Usage  # Token usage stats
    logprobs: Optional[Logprobs] = None  # Logprobs if requested
    echo: bool = False  # Whether echo mode was enabled

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (Usage, Logprobs)):
                result[key] = asdict(value)
            else:
                result[key] = value
        return result


@dataclass
class Rollouts:
    """Container for multiple LLM responses with metadata.

    This is the main dataclass returned by generate functions.
    It contains the prompt, configuration, and all generated responses.
    """

    prompt: str
    num_responses: int
    temperature: float
    top_p: float
    max_tokens: int
    provider: str
    model: str
    responses: List[FwResponse]
    cache_dir: str
    logprobs_enabled: bool = False
    echo_enabled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation (original format)."""
        return {
            "prompt": self.prompt,
            "num_responses": self.num_responses,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "provider": self.provider,
            "model": self.model,
            "responses": [r.to_dict() if hasattr(r, 'to_dict') else r for r in self.responses],
            "cache_dir": self.cache_dir,
            "logprobs_enabled": self.logprobs_enabled,
            "echo_enabled": self.echo_enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Rollouts":
        """Create Rollouts from dictionary (for loading cached data)."""
        # Convert responses list
        responses = []
        for resp_data in data.get("responses", []):
            # Convert nested dictionaries to dataclasses
            if "usage" in resp_data and isinstance(resp_data["usage"], dict):
                resp_data["usage"] = Usage(**resp_data["usage"])

            if "logprobs" in resp_data and isinstance(resp_data["logprobs"], dict):
                resp_data["logprobs"] = Logprobs(**resp_data["logprobs"])

            responses.append(FwResponse(**resp_data))

        # Create Rollouts with converted responses
        return cls(
            prompt=data["prompt"],
            num_responses=data["num_responses"],
            temperature=data["temperature"],
            top_p=data["top_p"],
            max_tokens=data["max_tokens"],
            provider=data["provider"],
            model=data["model"],
            responses=responses,
            cache_dir=data["cache_dir"],
            logprobs_enabled=data.get("logprobs_enabled", False),
            echo_enabled=data.get("echo_enabled", False),
        )

    def get_texts(self) -> List[str]:
        """Get just the generated texts from all responses."""
        return [r.text for r in self.responses]

    def get_full_texts(self) -> List[str]:
        """Get full texts (including prompt if echo) from all responses."""
        return [r.full_text for r in self.responses]

    def get_first_response(self) -> Optional[FwResponse]:
        """Get the first response if available."""
        return self.responses[0] if self.responses else None

    def has_logprobs(self) -> bool:
        """Check if any response has logprobs data."""
        return any(r.logprobs is not None for r in self.responses)

    def get_total_tokens_used(self) -> int:
        """Get total tokens used across all responses."""
        return sum(r.usage.total_tokens for r in self.responses if r.usage)

    def filter_by_finish_reason(self, reason: str) -> List[FwResponse]:
        """Get responses that finished with a specific reason."""
        return [r for r in self.responses if r.finish_reason == reason]

    def __len__(self) -> int:
        """Number of responses."""
        return len(self.responses)

    def __getitem__(self, index: int) -> FwResponse:
        """Get response by index."""
        return self.responses[index]

    def __iter__(self):
        """Iterate over responses."""
        return iter(self.responses)


# Helper function for backward compatibility
def dict_to_rollouts(data: Dict[str, Any]) -> Rollouts:
    """Convert a dictionary response to Rollouts dataclass.

    This is useful for converting existing cached responses.
    """
    return Rollouts.from_dict(data)


def rollouts_to_dict(rollouts: Rollouts) -> Dict[str, Any]:
    """Convert Rollouts dataclass back to dictionary.

    This maintains backward compatibility with code expecting dicts.
    """
    return rollouts.to_dict()
