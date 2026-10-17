import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import hashlib
import os
import json
import asyncio
import time
import httpx
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from dotenv import load_dotenv

from resample.provider_config import MODEL2PROVIDER, ProviderConfig
from resample.rollouts import Rollouts, FwResponse, Usage, Logprobs
from resample.rate_limiter import get_rate_limiter


"""
Together.ai API client with logprobs support and caching.

This module provides a drop-in replacement for fireworks_logprobs.py but uses Together.ai API
with full logprobs support including echo mode (getting logprobs for prompt tokens).

Key features:
- Full logprobs support with echo mode
- Automatic caching of responses (same structure as fireworks_logprobs.py)
- Concurrent request handling
- Exponential backoff retry logic
- Comprehensive error handling
- Async interface matching fireworks_logprobs.py

Main differences from fireworks_logprobs.py:
- Uses Together.ai completions API for echo mode support
- Returns logprobs for both prompt and generated tokens
- Maximum 5 alternatives per token (Together.ai limit)

Example usage:
    result = await call_generate_together(
        prompt="Your prompt here",
        num_responses=10,
        logprobs=True,
        echo=True
    )
"""

# Load environment variables
load_dotenv()

# Get Together API key
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    raise ValueError("Together API key not found")


async def make_together_request(
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    model: str,
    seed: int = None,
    max_retries: int = 3,
    verbose: bool = True,
    logprobs: int = 5,
    echo: bool = True,
    top_k: int = 40,
    presence_penalty: float = 0,
    frequency_penalty: float = 0,
) -> Dict:
    """Make an API request to Together.ai using completions format for logprobs support.

    Args:
        prompt: The prompt to send to the model
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens to generate
        model: Model name to use (will be converted to Together.ai format)
        seed: Random seed for reproducible generation
        max_retries: Maximum number of retries for failed requests
        verbose: Whether to print progress and status messages
        logprobs: Number of logprobs to return (max 5 for Together.ai)
        echo: Whether to include prompt tokens in response (for prompt logprobs)
        top_k: Top-k sampling parameter
        presence_penalty: Presence penalty
        frequency_penalty: Frequency penalty

    Returns:
        Dict containing the response with logprobs or error information
    """

    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json",
    }

    # Convert model names to Together.ai format
    # Together.ai uses different model naming conventions
    model_mapping = {
        "qwen-15b": "Qwen/Qwen2.5-1.5B-Instruct",
        "llama-v3p1-8b": "meta-llama/Llama-3.1-8B-Instruct",
        "gpt-oss-20b": "gpt-oss-20b",  # May need to find correct model name
        "qwen3-30b-a3b": "Qwen/Qwen2.5-32B-Instruct",  # Using closest available
        "qwen/qwen3-30b-a3b": "Qwen/Qwen2.5-32B-Instruct",
        "deepseek/deepseek-r1-distill-llama-70b": r"deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    }

    # Apply mapping if available, otherwise use as-is
    if model in model_mapping:
        model = model_mapping[model]
    elif "/" not in model:
        # If no org/model format and not in mapping, try to guess
        # This might need adjustment based on Together.ai's available models
        pass

    # Use completions API for logprobs and echo support
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "logprobs": logprobs,  # Number of logprobs (max 5)
        "echo": echo,  # Include prompt tokens for their logprobs
        "stream": False,
    }

    # Together.ai supports seed parameter
    if seed is not None:
        payload["seed"] = seed

    api_url = "https://api.together.xyz/v1/completions"

    # Implement exponential backoff for retries
    retry_delay = 2

    # Get the global rate limiter
    rate_limiter = get_rate_limiter()

    for attempt in range(max_retries):
        try:
            # Wait for rate limit permission before making the request
            await rate_limiter.acquire()

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    api_url, headers=headers, json=payload, timeout=300
                )

                if response.status_code == 500:
                    if verbose:
                        print(
                            f"Server error (500) on attempt {attempt+1}/{max_retries}. Retrying..."
                        )
                    delay = retry_delay * (2**attempt)
                    if delay > 1:
                        delay = 1
                    await asyncio.sleep(delay)
                    continue

                elif response.status_code == 429:
                    if verbose:
                        print(
                            f"Rate limit (429) on attempt {attempt+1}/{max_retries}. Retrying..."
                        )
                    delay = retry_delay * (2**attempt)
                    if delay > 1:
                        delay = 1
                    await asyncio.sleep(delay)
                    continue

                elif response.status_code != 200:
                    if verbose:
                        print(
                            f"Error from Together API: {response.status_code} - {response.text}"
                        )
                    if attempt == max_retries - 1:
                        if verbose:
                            print(f"Saving API error: {response.status_code}")
                        return {
                            "error": f"API error: {response.status_code}",
                            "details": response.text,
                        }
                    delay = retry_delay * (2**attempt)
                    if delay > 1:
                        delay = 1
                    await asyncio.sleep(delay)
                    continue

                result = response.json()

                # Extract response from Together.ai completions format
                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    text = choice.get("text", "")
                    finish_reason = choice.get("finish_reason", "")

                    # Extract logprobs data
                    logprobs_data = choice.get("logprobs", None)

                    # Handle echo mode - Together.ai returns prompt separately
                    prompt_tokens = []
                    prompt_token_logprobs = []
                    prompt_top_logprobs = []

                    if (
                        echo
                        and "prompt" in result
                        and len(result["prompt"]) > 0
                    ):
                        # Together.ai returns prompt tokens separately when echo=true
                        prompt_info = result["prompt"][0]
                        if "logprobs" in prompt_info:
                            prompt_lp = prompt_info["logprobs"]
                            prompt_tokens = prompt_lp.get("tokens", [])
                            prompt_token_logprobs = prompt_lp.get(
                                "token_logprobs", []
                            )
                            # Convert prompt top_logprobs if available
                            if "top_logprobs" in prompt_lp:
                                prompt_top_logprobs = prompt_lp.get(
                                    "top_logprobs", []
                                )

                    # Parse logprobs into structured format
                    structured_logprobs = None
                    if logprobs_data or prompt_tokens:
                        # Get generated tokens
                        gen_tokens = (
                            logprobs_data.get("tokens", [])
                            if logprobs_data
                            else []
                        )
                        gen_token_logprobs = (
                            logprobs_data.get("token_logprobs", [])
                            if logprobs_data
                            else []
                        )
                        gen_top_logprobs = (
                            logprobs_data.get("top_logprobs", [])
                            if logprobs_data
                            else []
                        )

                        # Combine prompt and generated tokens if echo is enabled
                        if echo and prompt_tokens:
                            tokens = prompt_tokens + gen_tokens
                            token_logprobs = (
                                prompt_token_logprobs + gen_token_logprobs
                            )

                            # Handle top_logprobs: prompt doesn't have them from Together.ai
                            # Create dicts with the actual token and ERROR marker for prompt tokens
                            if gen_top_logprobs or prompt_tokens:
                                # Create top_logprobs for prompt tokens with the actual token and ERROR marker
                                prompt_top_logprobs_filled = []
                                for i, token in enumerate(prompt_tokens):
                                    # Include the actual token with its logprob, and ERROR marker
                                    token_dict = {
                                        token: (
                                            prompt_token_logprobs[i]
                                            if prompt_token_logprobs[i]
                                            is not None
                                            else 0
                                        ),
                                        "ERROR": None,
                                    }
                                    prompt_top_logprobs_filled.append(
                                        token_dict
                                    )
                                top_logprobs = prompt_top_logprobs_filled + (
                                    gen_top_logprobs if gen_top_logprobs else []
                                )
                            else:
                                top_logprobs = []

                            # Create text_offset array (approximate since Together.ai doesn't provide it)
                            text_offset = []
                            offset = 0
                            for token in tokens:
                                text_offset.append(offset)
                                offset += len(token)
                        else:
                            tokens = gen_tokens
                            token_logprobs = gen_token_logprobs
                            top_logprobs = gen_top_logprobs
                            text_offset = []
                            offset = 0
                            for token in tokens:
                                text_offset.append(offset)
                                offset += len(token)

                        # Identify where prompt ends (if echo is enabled)
                        prompt_len = len(prompt) if echo else 0
                        num_prompt_tokens = len(prompt_tokens) if echo else 0

                        structured_logprobs = {
                            "tokens": tokens,
                            "token_logprobs": token_logprobs,
                            "top_logprobs": top_logprobs,
                            "text_offset": text_offset,
                            "prompt_length": prompt_len,
                            "num_prompt_tokens": num_prompt_tokens,
                        }

                    # The generated text is always just the completion
                    generated_text = text

                    # Construct full text if echo is enabled
                    if echo:
                        full_text = prompt + text
                    else:
                        full_text = text

                    response_data = {
                        "text": generated_text,  # Just the generated text
                        "full_text": full_text,  # Full text including prompt if echo
                        "post": generated_text,  # For compatibility with openrouter.py
                        "reasoning": "",  # Together.ai doesn't have separate reasoning
                        "finish_reason": finish_reason,
                        "provider": "Together",
                        "response_id": result.get("id", ""),
                        "model": model,
                        "object": result.get("object", ""),
                        "created": result.get("created", ""),
                        "usage": result.get("usage", {}),
                        "logprobs": structured_logprobs,
                        "echo": echo,
                    }

                    return response_data
                else:
                    ret_resp = {"error": "No choices in API response"}
                    for key in result:
                        ret_resp[key] = result[key]
                    if verbose:
                        print(f"No choices in API response: {ret_resp}")
                    return ret_resp

        except Exception as e:
            if verbose:
                print(
                    f"Exception during Together API request (attempt {attempt+1}/{max_retries}): {e}"
                )
            if attempt == max_retries - 1:
                return {"error": f"Request exception: {str(e)}"}
            delay = retry_delay * (2**attempt)
            if delay > 61:
                delay = 61
            await asyncio.sleep(delay)

    return {"error": "All API request attempts failed"}


async def generate_multiple_responses_together(
    prompt: str,
    num_responses: int,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 8192,
    model: str = "qwen3-30b-a3b",
    max_retries: int = 6,
    verbose: bool = False,
    logprobs: int = 5,
    echo: bool = True,
    top_k: int = 40,
    presence_penalty: float = 0,
    frequency_penalty: float = 0,
    two_layer_hash: bool = True,
    req_exist: bool = False,
    strict_req_post: bool = False,
    return_dataclass: bool = True,
) -> Union[Rollouts, Dict]:
    """
    Generate multiple responses for a given prompt using Together.ai with individual caching.

    This function maintains the same interface as fireworks_logprobs.py but uses Together.ai API
    with full logprobs support.

    Args:
        prompt: The prompt to send to the model
        num_responses: Number of responses to generate
        temperature: Sampling temperature (default: 0.6)
        top_p: Top-p sampling parameter (default: 0.95)
        max_tokens: Maximum tokens to generate (default: 16384)
        model: Model name to use (default: "qwen3-30b-a3b")
        max_retries: Maximum number of retries for failed requests (default: 6)
        verbose: Whether to print progress and status messages (default: False)
        logprobs: Number of logprobs to return, max 5 (default: 5)
        echo: Whether to include prompt tokens in logprobs (default: True)
        top_k: Top-k sampling parameter (default: 40)
        presence_penalty: Presence penalty (default: 0)
        frequency_penalty: Frequency penalty (default: 0)
        two_layer_hash: Whether to use two-layer directory structure for cache
        req_exist: Whether to only return existing cached responses
        strict_req_post: Whether to require 'post' field in cached responses

    Returns:
        Rollouts dataclass or dict containing responses with logprobs and metadata
    """

    # Create cache directory structure (compatible with fireworks_logprobs.py)
    model_str = model.replace("/", "-").replace(":", "")

    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    prompt_hash = prompt_hash[
        ::2
    ]  # Take every other character for shorter hash

    # Include logprobs and echo settings in cache key
    logprobs_str = f"_lp{logprobs}_echo{1 if echo else 0}"

    if two_layer_hash:
        cache_dir = f"response_cache_together/{model_str}/t{temperature}_p{top_p}_tok{max_tokens}{logprobs_str}_{prompt_hash[:3]}/{prompt_hash[3:]}"
    else:
        cache_dir = f"response_cache_together/{model_str}/t{temperature}_p{top_p}_tok{max_tokens}{logprobs_str}_{prompt_hash}"

    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Check API key
    if not TOGETHER_API_KEY:
        raise ValueError(
            "Together API key not found. Set TOGETHER_API_KEY environment variable or create together_api_key.txt"
        )

    # Check which seeds already exist
    existing_responses = {}
    missing_seeds = []

    for seed in range(num_responses):
        seed_file = f"{cache_dir}/seed_{seed:05d}.json"
        if os.path.exists(seed_file):
            try:
                with open(seed_file, "r") as f:
                    response_data = json.load(f)
                    if "error" in response_data:
                        missing_seeds.append(seed)
                        continue
                    # Check if logprobs are present when required
                    if logprobs > 0:
                        if (
                            "response" not in response_data
                            or "logprobs"
                            not in response_data.get("response", {})
                        ):
                            if verbose:
                                print(
                                    f"Missing logprobs for seed {seed}, will regenerate"
                                )
                            missing_seeds.append(seed)
                            continue
                    if strict_req_post:
                        if (
                            "post" not in response_data["response"]
                            or response_data["response"]["post"] == ""
                        ):
                            missing_seeds.append(seed)
                            continue
                    existing_responses[seed] = response_data
                    if verbose:
                        print(f"Found cached response for seed {seed}")
            except Exception as e:
                if verbose:
                    print(f"Error loading cached response for seed {seed}: {e}")
                missing_seeds.append(seed)
        else:
            missing_seeds.append(seed)

    if verbose:
        print(
            f"Found {len(existing_responses)} cached responses, need to generate {len(missing_seeds)} more"
        )

    if req_exist:
        all_responses = []
        for seed in range(num_responses):
            if seed in existing_responses:
                resp_data = existing_responses[seed]["response"]

                # Convert to Response dataclass if returning dataclass
                if return_dataclass and not isinstance(resp_data, FwResponse):
                    if "error" not in resp_data:
                        # Convert usage dict to Usage dataclass
                        usage_data = resp_data.get("usage", {})
                        usage_obj = Usage(
                            prompt_tokens=usage_data.get("prompt_tokens", 0),
                            completion_tokens=usage_data.get(
                                "completion_tokens", 0
                            ),
                            total_tokens=usage_data.get("total_tokens", 0),
                        )

                        # Convert logprobs dict to Logprobs dataclass
                        logprobs_obj = None
                        if resp_data.get("logprobs"):
                            lp_data = resp_data["logprobs"]
                            logprobs_obj = Logprobs(
                                tokens=lp_data.get("tokens", []),
                                token_logprobs=lp_data.get(
                                    "token_logprobs", []
                                ),
                                top_logprobs=lp_data.get("top_logprobs", []),
                                text_offset=lp_data.get("text_offset", []),
                                prompt_length=lp_data.get("prompt_length", 0),
                                num_prompt_tokens=lp_data.get(
                                    "num_prompt_tokens", 0
                                ),
                            )

                        # Create Response object
                        response_obj = FwResponse(
                            text=resp_data.get("text", ""),
                            full_text=resp_data.get("full_text", ""),
                            post=resp_data.get("post", ""),
                            reasoning=resp_data.get("reasoning", ""),
                            finish_reason=resp_data.get("finish_reason", ""),
                            provider=resp_data.get("provider", "Together"),
                            response_id=resp_data.get("response_id", ""),
                            model=resp_data.get("model", model),
                            object=resp_data.get("object", "text_completion"),
                            created=resp_data.get("created", 0),
                            usage=usage_obj,
                            logprobs=logprobs_obj,
                            echo=resp_data.get("echo", echo),
                        )
                        all_responses.append(response_obj)
                    else:
                        all_responses.append(resp_data)
                else:
                    all_responses.append(resp_data)

        # Create result as Rollouts or dict
        if return_dataclass:
            result = Rollouts(
                prompt=prompt,
                num_responses=num_responses,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                provider="Together",
                model=model,
                responses=all_responses,
                cache_dir=cache_dir,
                logprobs_enabled=logprobs > 0,
                echo_enabled=echo,
            )
        else:
            result = {
                "prompt": prompt,
                "num_responses": num_responses,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "provider": "Together",
                "model": model,
                "responses": all_responses,
                "cache_dir": cache_dir,
                "logprobs_enabled": logprobs > 0,
                "echo_enabled": echo,
            }
        return result

    # Create tasks for missing seeds only
    t_start = time.time()

    async def make_request_with_seed(seed):
        """Helper function that returns both seed and response"""
        response = await make_together_request(
            prompt,
            temperature,
            top_p,
            max_tokens,
            model,
            seed,
            max_retries,
            verbose,
            logprobs,
            echo,
            top_k,
            presence_penalty,
            frequency_penalty,
        )
        return seed, response

    tasks = [make_request_with_seed(seed) for seed in missing_seeds]

    # Execute requests and save individual responses as they complete
    if tasks:
        if verbose:
            print(
                f"Generating {len(tasks)} responses using Together with model {model}..."
            )

        completed_count = 0
        pbar = (
            tqdm(total=len(tasks), desc="Generating responses")
            if len(tasks) > 10
            else None
        )

        for task in asyncio.as_completed(tasks):
            seed, response = await task

            # Check if we need logprobs and didn't get them
            logprobs_retry_count = 0
            max_logprobs_retries = 2

            while logprobs > 0 and logprobs_retry_count < max_logprobs_retries:
                # Check if response has error or missing logprobs
                if "error" in response:
                    # Always print error messages
                    print(
                        f"\nWARNING: Seed {seed} returned error: {response.get('error', 'Unknown error')}"
                    )
                    break  # Don't retry errors
                elif response.get("logprobs") is None:
                    # Always print this warning
                    print(
                        f"\nWARNING: Seed {seed} missing logprobs, retrying... (attempt {logprobs_retry_count + 1}/{max_logprobs_retries})"
                    )

                    # Retry the request
                    logprobs_retry_count += 1
                    response = await make_together_request(
                        prompt,
                        temperature,
                        top_p,
                        max_tokens,
                        model,
                        seed,
                        max_retries,
                        verbose,
                        logprobs,
                        echo,
                        top_k,
                        presence_penalty,
                        frequency_penalty,
                    )
                else:
                    # We have logprobs, break out of retry loop
                    if logprobs_retry_count > 0:
                        print(
                            f"\nSUCCESS: Seed {seed} now has logprobs after {logprobs_retry_count} retries"
                        )
                    break

            # If we still don't have logprobs after retries, warn
            if (
                logprobs > 0
                and "error" not in response
                and response.get("logprobs") is None
            ):
                print(
                    f"\nERROR: Seed {seed} still missing logprobs after {max_logprobs_retries} retries!"
                )

            # Save individual response immediately
            seed_file = f"{cache_dir}/seed_{seed:05d}.json"
            response_data = {
                "seed": seed,
                "prompt": prompt,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "model": model,
                "logprobs": logprobs,
                "echo": echo,
                "response": response,
            }

            try:
                with open(seed_file, "w", encoding="utf-8") as f:
                    json.dump(response_data, f, indent=2)
                existing_responses[seed] = response_data
                if verbose and "text" in response:
                    print(f"\nSaved response for seed {seed}")
            except Exception as e:
                if verbose:
                    print(f"\nError saving response for seed {seed}: {e}")

            completed_count += 1
            if verbose and pbar:
                pbar.update(1)

        if verbose and pbar:
            pbar.close()

    # Compile all responses in order
    all_responses = []
    for seed in range(num_responses):
        if seed in existing_responses:
            resp_data = existing_responses[seed]["response"]

            # Convert to Response dataclass if returning dataclass
            if return_dataclass and not isinstance(resp_data, FwResponse):
                # Handle error responses
                if "error" in resp_data:
                    all_responses.append(resp_data)  # Keep error as dict
                else:
                    # Convert usage dict to Usage dataclass
                    usage_data = resp_data.get("usage", {})
                    usage_obj = Usage(
                        prompt_tokens=usage_data.get("prompt_tokens", 0),
                        completion_tokens=usage_data.get(
                            "completion_tokens", 0
                        ),
                        total_tokens=usage_data.get("total_tokens", 0),
                    )

                    # Convert logprobs dict to Logprobs dataclass
                    logprobs_obj = None
                    if resp_data.get("logprobs"):
                        lp_data = resp_data["logprobs"]
                        logprobs_obj = Logprobs(
                            tokens=lp_data.get("tokens", []),
                            token_logprobs=lp_data.get("token_logprobs", []),
                            top_logprobs=lp_data.get("top_logprobs", []),
                            text_offset=lp_data.get("text_offset", []),
                            prompt_length=lp_data.get("prompt_length", 0),
                            num_prompt_tokens=lp_data.get(
                                "num_prompt_tokens", 0
                            ),
                        )

                    # Create Response object
                    response_obj = FwResponse(
                        text=resp_data.get("text", ""),
                        full_text=resp_data.get("full_text", ""),
                        post=resp_data.get("post", ""),
                        reasoning=resp_data.get("reasoning", ""),
                        finish_reason=resp_data.get("finish_reason", ""),
                        provider=resp_data.get("provider", "Together"),
                        response_id=resp_data.get("response_id", ""),
                        model=resp_data.get("model", model),
                        object=resp_data.get("object", "text_completion"),
                        created=resp_data.get("created", 0),
                        usage=usage_obj,
                        logprobs=logprobs_obj,
                        echo=resp_data.get("echo", echo),
                    )
                    all_responses.append(response_obj)
            else:
                all_responses.append(resp_data)
        else:
            all_responses.append({"error": f"Missing response for seed {seed}"})

    # Create result as Rollouts or dict
    if return_dataclass:
        result = Rollouts(
            prompt=prompt,
            num_responses=num_responses,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            provider="Together",
            model=model,
            responses=all_responses,
            cache_dir=cache_dir,
            logprobs_enabled=logprobs > 0,
            echo_enabled=echo,
        )
        # Convert to dict for saving
        result_data = result.to_dict()
    else:
        # Return as dict for backward compatibility
        result_data = {
            "prompt": prompt,
            "num_responses": num_responses,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "provider": "Together",
            "model": model,
            "responses": all_responses,
            "cache_dir": cache_dir,
            "logprobs_enabled": logprobs > 0,
            "echo_enabled": echo,
        }
        result = result_data

    # Also save consolidated file for compatibility
    consolidated_file = f"{cache_dir}/consolidated.json"
    with open(consolidated_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    if verbose:
        print(f"\nCompleted all {num_responses} responses")
        print(f"Individual responses saved in: {cache_dir}")
        print(f"Time taken: {time.time() - t_start:.2f} seconds")

    return result


# New function to work with ProviderConfig
async def gen_with_config(
    prompt: str,
    num_responses: int,
    config: Union[ProviderConfig, Dict[str, Any], None] = None,
    return_dataclass: bool = True,
    **override_kwargs,
) -> Union[Rollouts, Dict]:
    """Generate responses using a ProviderConfig object.

    Args:
        prompt: The prompt to send to the model
        num_responses: Number of responses to generate
        config: ProviderConfig object, dict, or None (uses defaults)
        return_dataclass: Whether to return Rollouts dataclass (True) or dict (False)
        **override_kwargs: Any parameters to override from config

    Returns:
        Rollouts dataclass or dict containing responses and metadata

    Example:
        # Use default config
        result = await generate_with_config(prompt, 5)

        # Use custom config
        config = ProviderConfig(temperature=0.9, max_tokens=4096)
        result = await generate_with_config(prompt, 5, config)

        # Use config with overrides
        config = ProviderConfig()
        result = await generate_with_config(prompt, 5, config, temperature=1.2)
    """

    # Handle different config input types
    if config is None:
        # Use defaults
        params = {}
    elif hasattr(config, "to_api_params") and callable(config.to_api_params):
        # Use ProviderConfig object (check by duck typing instead of isinstance)
        # This handles both direct imports and module-qualified imports
        params = config.to_api_params(exclude=["max_retries", "verbose"])
        # Extract non-API params separately
        params["max_retries"] = config.max_retries
        params["verbose"] = config.verbose
    elif isinstance(config, dict):
        # Use dictionary
        params = config.copy()
    else:
        raise ValueError(
            f"Config must be ProviderConfig, dict, or None, got {type(config)}"
        )

    assert (
        MODEL2PROVIDER[config.model] == "together"
    ), f"Bad model in config: {config.model=}"

    # Apply any overrides
    params.update(override_kwargs)
    del params["provider"]

    # Add return_dataclass to params if not already in override_kwargs
    if "return_dataclass" not in params:
        params["return_dataclass"] = return_dataclass

    # Generate responses
    return await generate_multiple_responses_together(
        prompt=prompt, num_responses=num_responses, **params
    )
