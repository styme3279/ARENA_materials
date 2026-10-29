# %%

import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from IPython.display import display
from openai import OpenAI
from scipy import stats

# Make sure exercises are in the path
chapter = "chapter4_alignment_science"
section = "part5_investigator_agents"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section

if str(exercises_dir) not in sys.path:
    sys.path.insert(0, str(exercises_dir))

import part5_investigator_agents.tests as tests
import part5_investigator_agents.utils as utils

MAIN = __name__ == "__main__"

# %%
# Load environment variables
env_path = exercises_dir / ".env"
assert env_path.exists(), "Please create a .env file with: OPENROUTER_API_KEY=sk-or-..."

load_dotenv(dotenv_path=str(env_path))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
assert OPENROUTER_API_KEY, "Please set OPENROUTER_API_KEY in your .env file"

# OpenRouter client (used by generate_response throughout all sections)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# %%

# Load environment variables
env_path = exercises_dir / ".env"
assert env_path.exists(), "Please create a .env file with: OPENROUTER_API_KEY=sk-or-..."

load_dotenv(dotenv_path=str(env_path))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
assert OPENROUTER_API_KEY, "Please set OPENROUTER_API_KEY in your .env file"

# OpenRouter client (used by generate_response throughout all sections)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# %%

# Model constants - used by generate_response (OpenRouter format: "provider/model")
AUDITOR_MODEL = "anthropic/claude-haiku-5.5"  # For running investigations
TARGET_MODEL = "deepseek/deepseek-chat"  # Target to probe (found vulnerable in AI psychosis study)
JUDGE_MODEL = "anthropic/claude-haiku-5.5"  # For grading responses

# inspect_ai model names - used by Petri audit (Section 2 only)
# All routed through OpenRouter so we only need one API key.
# Format: "openrouter/<openrouter_model_id>"
INSPECT_AUDITOR_MODEL = "openrouter/anthropic/claude-haiku-5.5"
INSPECT_TARGET_MODEL = "openrouter/deepseek/deepseek-chat"
INSPECT_JUDGE_MODEL = "openrouter/anthropic/claude-haiku-5.5"


def generate_response(
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 2000,
    stop_sequences: list[str] | None = None,
    temperature: float = 0.7,
    max_retries: int = 10,
) -> str:
    """Single API call with retry logic for rate limits."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop_sequences if stop_sequences else None,
            )
            if not response.choices:
                raise ValueError("API returned empty/null choices")
            return response.choices[0].message.content or ""
        except Exception as e:
            print(str(e))
            if any(msg in str(e) for msg in ("rate_limit", "429", "empty/null choices")):
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    print(f"Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
            raise e
    return ""
