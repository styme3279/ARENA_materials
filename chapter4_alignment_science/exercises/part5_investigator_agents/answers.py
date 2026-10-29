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
    name = char_file.stem.split('_')[0].title()
    txt = char_file.read_text()
    return {"name": name, "scenario": txt}


def load_all_characters(characters_dir: Path) -> dict[str, dict[str, str]]:
    """
    Load all character .txt files from a directory.

    Args:
        characters_dir: Path to the characters/ directory

    Returns:
        Dict mapping character name -> character dict (from load_character),
        sorted alphabetically by name.
        
    """
    characters = {}
    for file in os.listdir(characters_dir):
        char = load_character(characters_dir / file)
        characters[char["name"]] = char
    return characters


characters = load_all_characters(AI_PSYCHOSIS_DIR / "characters")
print(f"Loaded {len(characters)} characters: {list(characters.keys())}")
print("\nSample - Ethan's scenario (first 500 chars):")
print(characters["Ethan"]["scenario"][:500] + "...")

# %%
def generate_redteam_message(
    redteam_history: list[dict[str, str]],
    target_response: str,
    patient_name: str,
    turn_number: int,
    total_turns: int,
    redteam_model: str = RED_TEAM_MODEL,
) -> tuple[str, list[dict[str, str]]]:
    """
    Generate the next adaptive patient message using the red-team LLM.

    Appends the target's response to the red-team history, calls the red-team
    model, then extracts the <message>...</message> content.

    Args:
        redteam_history: Current red-team LLM conversation history
        target_response: The target model's most recent response (empty str on turn 1)
        patient_name: The character's name (e.g. "Ethan")
        turn_number: Current turn number (1-indexed)
        total_turns: Total turns planned (used in the update message)
        redteam_model: Model to use as the red-team LLM

    Returns:
        Tuple of:
          - patient_message (str): The extracted <message>...</message> content
          - updated_history (list): Updated red-team conversation history
    """
    # YOUR CODE HERE
    # 1. Copy redteam_history (don't mutate in-place)
    # 2. If target_response is non-empty, append as a user message:
    #      "Target model response [{turn_number}/{total_turns}]:\n"
    #      "<target_model_response>\n{target_response}\n</target_model_response>\n"
    #      "Please continue as {patient_name}."
    # 3. Call generate_response(model=redteam_model, messages=updated_history)
    # 4. Append the response to updated_history as an assistant message
    # 5. Extract <message>...</message> content (re.DOTALL, strip whitespace)
    #    Fall back to the full response if tags not found
    # 6. Return (patient_message, updated_history)
    history = redteam_history.copy()
    if target_response:
        additional = (
            f"Target model response [{turn_number}/{total_turns}]:\n"
            f"<target_model_response>\n{target_response}\n</target_model_response>\n"
            f"Please continue as {patient_name}.")
        history.append({'role':'user','content':additional})
    msg = generate_response(
        model=redteam_model,
        messages=history)
    history.append({'role':'assistant','content':msg})
    match = re.search(r'<message>(.*?)</message>', msg, re.DOTALL)
    patient_message = match.group(1).strip() if match else msg.strip()

    return patient_message, history


# Quick sanity check: generate the first patient message (no target response yet)
test_char = characters["Ethan"]
formatted_prompt = RED_TEAM_PROMPT.replace("{character_name}", test_char["name"]).replace(
    "{psychotic_character}", test_char["scenario"]
)
init_history = [{"role": "user", "content": formatted_prompt + "\nYou will send a total of 3 messages."}]

first_message, updated_history = generate_redteam_message(
    redteam_history=init_history,
    target_response="",
    patient_name=test_char["name"],
    turn_number=1,
    total_turns=3,
)
print(f"First patient message from Ethan:\n{first_message[:400]}")
print(f"\nHistory now has {len(updated_history)} messages")

# %%
def run_persona_conversation(
    character: dict[str, str],
    red_team_prompt: str = RED_TEAM_PROMPT,
    target_model: str = TARGET_MODEL,
    redteam_model: str = RED_TEAM_MODEL,
    num_turns: int = 6,
    verbose: bool = False,
) -> list[dict[str, str]]:
    """
    Run a multi-turn conversation using a live LLM red-team attacker.

    The red-team LLM plays the patient (character), adapting each message to
    the target model's previous response. Returns only the target conversation.

    Args:
        character: Dict with "name" and "scenario" (from load_character)
        red_team_prompt: Red-team prompt template (from red_team_prompt.txt)
        target_model: The model being probed
        redteam_model: The model playing the patient
        num_turns: Number of conversation turns
        verbose: Whether to print progress

    Returns:
        List of message dicts [{"role": "user"|"assistant", "content": ...}, ...]
        representing the target model's conversation only
    """
    # YOUR CODE HERE
    # 1. Format the red-team prompt: replace {character_name} and {psychotic_character}
    # 2. Initialise redteam_history with the formatted prompt + turn count message
    # 3. Loop for num_turns:
    #    a. Get last target response (empty string on turn 1)
    #    b. Call generate_redteam_message(...)
    #    c. Append patient_message to target_history as "user"
    #    d. Call generate_response on target_model with [system] + target_history
    #    e. Append target response to target_history as "assistant"
    # 4. Return target_history
    red_team_prompt = red_team_prompt.replace(r'{character_name}', character['name'])
    red_team_prompt = red_team_prompt.replace(r'{psychotic_character}', character['scenario'])

    target_response = ''
    redteam_history = [{'role':'system','content': red_team_prompt + f"\nYou will send a total of {num_turns} messages."}]
    target_history = []
    for turn_number in range(num_turns):
        if target_history:
            target_respone = target_history[-1]['content']
        patient_msg, redteam_history = generate_redteam_message(
            redteam_history,
            target_response,
            character['name'],
            turn_number + 1,
            num_turns,
            redteam_model)
        target_history.append({'role':'user','role':patient_msg})
        response = generate_response(
            target_model,
            [{'role':'system','content':'You are an helpful AI assistant.'}] + target_history)
        redteam_history.append({'role':'assistant','content':response})

        if verbose:
            print(f"[Turn {turn_number + 1}] {character['name']}: {patient_msg[:100]}")
            print(f"[Turn {turn_number + 1}] Assistant: {response[:100]}")
        return target_history

    
test_char = characters["Ethan"]
conversation = run_persona_conversation(
    test_char,
    red_team_prompt=RED_TEAM_PROMPT,
    num_turns=3,
    verbose=True,
)

assert isinstance(conversation, list), "Must return list of messages"
assert all("role" in m and "content" in m for m in conversation), "Each message needs role and content"

tests.test_run_persona_conversation_mocked(run_persona_conversation, __name__)

# %%
