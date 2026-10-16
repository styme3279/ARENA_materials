"""
Global rate limiter for API calls.

This module provides a singleton rate limiter that can be shared across
all async operations to ensure we stay within API rate limits.
"""

import asyncio
import time
from typing import Optional


class RateLimiter:
    """A token bucket rate limiter for controlling API request rates."""

    def __init__(self, requests_per_minute: int = 550):
        """
        Initialize the rate limiter.

        Args:
            requests_per_minute: Maximum number of requests per minute
        """
        self.rpm = requests_per_minute
        self.requests_per_second = requests_per_minute / 60.0
        self.min_interval = 1.0 / self.requests_per_second

        # Token bucket parameters
        self.max_tokens = min(10, requests_per_minute // 6)  # Max burst size
        self.tokens = self.max_tokens
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """
        Acquire permission to make a request. This will wait if necessary
        to maintain the rate limit.
        """
        async with self.lock:
            while True:
                now = time.monotonic()
                time_passed = now - self.last_update

                # Add tokens based on time passed
                self.tokens = min(
                    self.max_tokens,
                    self.tokens + time_passed * self.requests_per_second,
                )
                self.last_update = now

                if self.tokens >= 1:
                    self.tokens -= 1
                    return

                # Calculate wait time until we have a token
                wait_time = (1 - self.tokens) / self.requests_per_second
                await asyncio.sleep(wait_time)

    def reset(self, requests_per_minute: int):
        """Reset the rate limiter with a new RPM limit."""
        self.__init__(requests_per_minute)


# Global singleton instance
_global_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter(requests_per_minute: int = 550) -> RateLimiter:
    """
    Get or create the global rate limiter singleton.

    Args:
        requests_per_minute: RPM limit (only used on first call)

    Returns:
        The global RateLimiter instance
    """
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = RateLimiter(requests_per_minute)
    return _global_rate_limiter


def set_rate_limit(requests_per_minute: int):
    """
    Set or update the global rate limit.

    Args:
        requests_per_minute: New RPM limit
    """
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = RateLimiter(requests_per_minute)
    else:
        _global_rate_limiter.reset(requests_per_minute)

    print(
        f"Rate limit set to {requests_per_minute} requests per minute ({requests_per_minute/60:.1f} RPS)"
    )
