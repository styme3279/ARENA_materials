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

# %%
# Model to use as the red-team LLM (playing the patient).
# The original study used grok-3, which has since been deprecated. grok-4.5 commits to the personas; grok-4.3,
# grok-4.20 and grok-4.7 refuse to play the patient (and the refusal then gets sent to the target as if it were
# the patient's message), so if you swap models, read the first few patient messages to check it's in character.
# Warning: using Claude as the red-team model can trigger account bans at scale
# due to the nature of the content being generated!
RED_TEAM_MODEL = "x-ai/grok-4.5"  # via OpenRouter

# Check every model we use is still served by OpenRouter. Models get deprecated regularly, and a dead model
# otherwise shows up much later as empty transcripts or all-zero scores rather than as an error.
for _model in {AUDITOR_MODEL, TARGET_MODEL, JUDGE_MODEL, RED_TEAM_MODEL}:
    try:
        client.chat.completions.create(model=_model, messages=[{"role": "user", "content": "hi"}], max_tokens=1)
    except Exception as e:
        raise RuntimeError(f"Model {_model!r} is unavailable on OpenRouter - swap it for a current model: {e}")

# %%

# Path to the ai-psychosis repo (already cloned in the ARENA environment)
AI_PSYCHOSIS_DIR = exercises_dir / "ai-psychosis"
assert AI_PSYCHOSIS_DIR.exists(), f"ai-psychosis repo not found at {AI_PSYCHOSIS_DIR}"

# Load the red-team and grader prompts directly from the repo
RED_TEAM_PROMPT = (AI_PSYCHOSIS_DIR / "red_team_prompt.txt").read_text()
GRADER_PROMPT = (AI_PSYCHOSIS_DIR / "grader_prompt.txt").read_text()

print(f"Loaded red-team prompt ({len(RED_TEAM_PROMPT)} chars)")
print(f"Loaded grader prompt ({len(GRADER_PROMPT)} chars)")

# %%
def load_character(char_file: Path) -> dict[str, str]:
    """
    Load a single character file from the ai-psychosis repo.

    Args:
        char_file: Path to a .txt character file (e.g. "ethan_reality.txt")

    Returns:
        Dict with:
          - "name": character's first name, title-cased (e.g. "Ethan")
          - "scenario": full file text, passed verbatim to the red-team LLM
    """
    # YOUR CODE HERE
    raise NotImplementedError()


def load_all_characters(characters_dir: Path) -> dict[str, dict[str, str]]:
    """
    Load all character .txt files from a directory.

    Args:
        characters_dir: Path to the characters/ directory

    Returns:
        Dict mapping character name -> character dict (from load_character),
        sorted alphabetically by name.
    """
    # YOUR CODE HERE
    res = {}
    for file in os.list(characters_dir):
        res['name'] = file.split('_')[0]
        res['scenario'] = file.split('_')[1]
    return res


characters = load_all_characters(AI_PSYCHOSIS_DIR / "characters")
print(f"Loaded {len(characters)} characters: {list(characters.keys())}")
print("\nSample - Ethan's scenario (first 500 chars):")
print(characters["Ethan"]["scenario"][:500] + "...")
