#!/usr/bin/env python3
"""
Clean OpenRouter API client using dataclasses for structured responses.

This module provides a clean interface to OpenRouter that returns
Rollouts and FwResponse dataclasses, matching the interface of
fireworks_logprobs.py for consistency.
"""

import hashlib
import os
import json
import asyncio
import time
import httpx
from pathlib import Path
from typing import List, Dict, Union, Any, Optional
from dotenv import load_dotenv

# Import dataclasses
from .rollouts import Rollouts, FwResponse, Usage

# Load environment variables
load_dotenv()

# Get OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OpenRouter API key not found")

async def make_openrouter_request(
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    model: str,
    seed: int = None,
    max_retries: int = 3,
    verbose: bool = False,
) -> FwResponse:
    """
    Make a single request to OpenRouter and return FwResponse dataclass.

    Args:
        prompt: The prompt to send to the model
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens to generate
        model: Model name to use
        seed: Random seed for reproducible generation
        max_retries: Maximum number of retries for failed requests
        verbose: Whether to print progress and status messages

    Returns:
        FwResponse dataclass with the response data
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://mmlu-reasoning-experiment",
        "X-Title": "MMLU Reasoning Research",
    }

    # Format as messages for the API (OpenRouter only supports chat completions)
    messages = [{"role": "user", "content": prompt}]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }

    # Add seed if provided
    if seed is not None:
        payload["seed"] = seed

    api_url = "https://openrouter.ai/api/v1/chat/completions"

    # Implement exponential backoff for retries
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    api_url, headers=headers, json=payload, timeout=300
                )

                if response.status_code in [500, 429]:
                    if verbose:
                        error_type = (
                            "Server error (500)"
                            if response.status_code == 500
                            else "Rate limit (429)"
                        )
                        print(
                            f"{error_type} on attempt {attempt+1}/{max_retries}. Retrying..."
                        )
                    delay = min(retry_delay * (2**attempt), 61)
                    await asyncio.sleep(delay)
                    continue

                elif response.status_code != 200:
                    if verbose:
                        print(
                            f"Error from OpenRouter API: {response.status_code} - {response.text}"
                        )
                    if attempt == max_retries - 1:
                        # Return error response
                        return FwResponse(
                            text=f"API error: {response.status_code}",
                            full_text=f"API error: {response.status_code}",
                            post="",
                            reasoning="",
                            finish_reason="error",
                            provider="OpenRouter",
                            response_id="",
                            model=model,
                            object="error",
                            created=int(time.time()),
                            usage=Usage(
                                prompt_tokens=0,
                                completion_tokens=0,
                                total_tokens=0,
                            ),
                            logprobs=None,
                            echo=False,
                        )
                    delay = min(retry_delay * (2**attempt), 61)
                    await asyncio.sleep(delay)
                    continue

                result = response.json()

                # Extract response from OpenRouter format
                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    message = choice.get("message", {})

                    # Handle the reasoning/content split
                    # OpenRouter returns thinking in "reasoning" and final answer in "content"
                    reasoning_text = message.get("reasoning", "")
                    content_text = message.get("content", "")

                    # Combine them appropriately
                    if reasoning_text and content_text:
                        # Both present - typical thinking mode response
                        full_text = (
                            f"{reasoning_text}\n</think>\n{content_text}"
                        )
                    elif reasoning_text:
                        # Only reasoning - model put everything in reasoning field
                        full_text = reasoning_text
                        if "</think>" not in full_text:
                            full_text += "\n</think>\n"
                    else:
                        # Only content - non-thinking mode
                        full_text = content_text

                    # The "text" field should be just the generated text (no prompt)
                    # Since echo=False for OpenRouter, text equals full_text
                    text = full_text

                    # Extract usage stats
                    usage_data = result.get("usage", {})
                    usage = Usage(
                        prompt_tokens=usage_data.get("prompt_tokens", 0),
                        completion_tokens=usage_data.get(
                            "completion_tokens", 0
                        ),
                        total_tokens=usage_data.get("total_tokens", 0),
                    )

                    # Create FwResponse
                    return FwResponse(
                        text=text,
                        full_text=full_text,
                        post=content_text,  # The final answer part
                        reasoning=reasoning_text,  # The thinking part
                        finish_reason=choice.get("finish_reason", ""),
                        provider=result.get("provider", "OpenRouter"),
                        response_id=result.get("id", ""),
                        model=result.get("model", model),
                        object=result.get("object", "chat.completion"),
                        created=result.get("created", int(time.time())),
                        usage=usage,
                        logprobs=None,  # OpenRouter doesn't provide logprobs
                        echo=False,  # Always False for OpenRouter
                    )
                else:
                    if verbose:
                        print(f"No choices in API response: {result}")
                    return FwResponse(
                        text="No response generated",
                        full_text="No response generated",
                        post="",
                        reasoning="",
                        finish_reason="error",
                        provider="OpenRouter",
                        response_id="",
                        model=model,
                        object="error",
                        created=int(time.time()),
                        usage=Usage(
                            prompt_tokens=0, completion_tokens=0, total_tokens=0
                        ),
                        logprobs=None,
                        echo=False,
                    )

        except Exception as e:
            if verbose:
                print(
                    f"Exception during OpenRouter API request (attempt {attempt+1}/{max_retries}): {e}"
                )
            if attempt == max_retries - 1:
                return FwResponse(
                    text=f"Request exception: {str(e)}",
                    full_text=f"Request exception: {str(e)}",
                    post="",
                    reasoning="",
                    finish_reason="error",
                    provider="OpenRouter",
                    response_id="",
                    model=model,
                    object="error",
                    created=int(time.time()),
                    usage=Usage(
                        prompt_tokens=0, completion_tokens=0, total_tokens=0
                    ),
                    logprobs=None,
                    echo=False,
                )
            delay = min(retry_delay * (2**attempt), 61)
            await asyncio.sleep(delay)

    # Should not reach here
    return FwResponse(
        text="All API request attempts failed",
        full_text="All API request attempts failed",
        post="",
        reasoning="",
        finish_reason="error",
        provider="OpenRouter",
        response_id="",
        model=model,
        object="error",
        created=int(time.time()),
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        logprobs=None,
        echo=False,
    )


async def generate_responses_openrouter(
    prompt: str,
    num_responses: int,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 16384,
    model: str = "qwen/qwen3-30b-a3b",
    max_retries: int = 3,
    verbose: bool = False,
) -> Rollouts:
    """
    Generate multiple responses using OpenRouter and return Rollouts dataclass.

    Args:
        prompt: The prompt to send to the model
        num_responses: Number of responses to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens to generate
        model: Model name to use
        max_retries: Maximum number of retries for failed requests
        verbose: Whether to print progress and status messages

    Returns:
        Rollouts dataclass containing all responses and metadata
    """

    # Check API key
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable or create openrouter_api_key.txt"
        )

    # Create cache directory structure
    model_str = model.replace("/", "-").replace(":", "")
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:32]

    cache_dir = f"response_cache_openrouter_clean/{model_str}/t{temperature}_p{top_p}_tok{max_tokens}/{prompt_hash}"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Check which seeds already exist in cache
    responses = []
    missing_seeds = []

    for seed in range(num_responses):
        seed_file = f"{cache_dir}/seed_{seed:05d}.json"
        if os.path.exists(seed_file):
            try:
                with open(seed_file, "r") as f:
                    cached_data = json.load(f)

                    # Convert cached data to FwResponse
                    if (
                        "response" in cached_data
                        and "error" not in cached_data["response"]
                    ):
                        resp_data = cached_data["response"]

                        # Create Usage object
                        usage = Usage(
                            prompt_tokens=resp_data.get("usage", {}).get(
                                "prompt_tokens", 0
                            ),
                            completion_tokens=resp_data.get("usage", {}).get(
                                "completion_tokens", 0
                            ),
                            total_tokens=resp_data.get("usage", {}).get(
                                "total_tokens", 0
                            ),
                        )

                        # Create FwResponse from cached data
                        fw_response = FwResponse(
                            text=resp_data.get("text", ""),
                            full_text=resp_data.get(
                                "text", ""
                            ),  # Same as text since echo=False
                            post=resp_data.get("post", ""),
                            reasoning=resp_data.get("reasoning", ""),
                            finish_reason=resp_data.get("finish_reason", ""),
                            provider=resp_data.get("provider", "OpenRouter"),
                            response_id=resp_data.get("response_id", ""),
                            model=resp_data.get("model", model),
                            object=resp_data.get("object", "chat.completion"),
                            created=resp_data.get("created", 0),
                            usage=usage,
                            logprobs=None,
                            echo=False,
                        )

                        responses.append(fw_response)
                        if verbose:
                            print(f"Found cached response for seed {seed}")
                    else:
                        missing_seeds.append(seed)
            except Exception as e:
                if verbose:
                    print(f"Error loading cached response for seed {seed}: {e}")
                missing_seeds.append(seed)
        else:
            missing_seeds.append(seed)

    if verbose and missing_seeds:
        print(
            f"Found {len(responses)} cached responses, generating {len(missing_seeds)} more"
        )

    # Generate missing responses
    if missing_seeds:
        tasks = []
        for seed in missing_seeds:
            task = make_openrouter_request(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                model=model,
                seed=seed,
                max_retries=max_retries,
                verbose=verbose,
            )
            tasks.append((seed, task))

        # Execute requests concurrently
        for seed, task in tasks:
            fw_response = await task

            # Save to cache
            seed_file = f"{cache_dir}/seed_{seed:05d}.json"
            cache_data = {
                "seed": seed,
                "prompt": prompt,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "model": model,
                "response": fw_response.to_dict(),
            }

            try:
                with open(seed_file, "w", encoding="utf-8") as f:
                    json.dump(cache_data, f, indent=2)
                if verbose:
                    print(f"Saved response for seed {seed}")
            except Exception as e:
                if verbose:
                    print(f"Error saving response for seed {seed}: {e}")

            # Add to responses list
            if fw_response.finish_reason != "error":
                responses.append(fw_response)

    # Create and return Rollouts
    rollouts = Rollouts(
        prompt=prompt,
        num_responses=num_responses,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        provider="OpenRouter",
        model=model,
        responses=responses,
        cache_dir=cache_dir,
        logprobs_enabled=False,  # Always False for OpenRouter
        echo_enabled=False,  # Always False for OpenRouter
    )

    if verbose:
        print(f"Generated {len(rollouts.responses)} responses successfully")

    return rollouts


# Backward compatibility function (matching old interface)
async def generate_multiple_responses_openrouter(
    prompt: str,
    num_responses: int,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 16384,
    model: str = "qwen/qwen3-30b-a3b",
    max_retries: int = 6,
    verbose: bool = False,
    **kwargs,  # Ignore extra parameters for compatibility
) -> Dict[str, Any]:
    """
    Generate multiple responses (backward compatibility wrapper).

    This function maintains compatibility with the old interface that
    returned dictionaries instead of dataclasses.
    """
    rollouts = await generate_responses_openrouter(
        prompt=prompt,
        num_responses=num_responses,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        model=model,
        max_retries=max_retries,
        verbose=verbose,
    )

    # Convert to dictionary format for backward compatibility
    return rollouts.to_dict()


# Main entry point matching fireworks_logprobs.py interface
async def gen_with_config(
    prompt: str,
    num_responses: int,
    config: Optional[Any] = None,
    return_dataclass: bool = True,
    **override_kwargs,
) -> Union[Rollouts, Dict]:
    """
    Generate responses using a config object (matches fireworks_logprobs interface).

    Args:
        prompt: The prompt to send to the model
        num_responses: Number of responses to generate
        config: Optional configuration object (supports dict or object with attributes)
        return_dataclass: Whether to return Rollouts dataclass (True) or dict (False)
        **override_kwargs: Any parameters to override from config

    Returns:
        Rollouts dataclass or dict containing responses and metadata
    """
    # Extract parameters from config
    params = {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 16384,
        "model": "qwen/qwen3-30b-a3b",
        "max_retries": 3,
        "verbose": False,
    }

    if config is not None:
        if hasattr(config, "__dict__"):
            # Object with attributes
            for key in params:
                if hasattr(config, key):
                    params[key] = getattr(config, key)
        elif isinstance(config, dict):
            # Dictionary
            for key in params:
                if key in config:
                    params[key] = config[key]

    # Apply overrides
    params.update(override_kwargs)

    # Generate responses
    rollouts = await generate_responses_openrouter(
        prompt=prompt, num_responses=num_responses, **params
    )

    if return_dataclass:
        return rollouts
    else:
        return rollouts.to_dict()


# Example usage
if __name__ == "__main__":

    async def test_clean_openrouter():
        print("Testing clean OpenRouter API with dataclasses...")

        prompt = "What is 2+2? Answer: "

        # Test with Rollouts return
        rollouts = await generate_responses_openrouter(
            prompt=prompt,
            num_responses=2,
            temperature=0.6,
            max_tokens=10,
            model="qwen/qwen3-30b-a3b",
            verbose=True,
        )

        print(f"\nPrompt: {rollouts.prompt}")
        print(f"Model: {rollouts.model}")
        print(f"Responses generated: {len(rollouts)}")

        for i, response in enumerate(rollouts):
            print(f"\nResponse {i+1}:")
            print(f"  Text: {response.text}")
            print(f"  Reasoning: {response.reasoning}")
            print(f"  Post: {response.post}")
            print(f"  Tokens: {response.usage.total_tokens}")

    asyncio.run(test_clean_openrouter())
