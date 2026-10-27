# %%
import os
import random
import re
import sys
from functools import partial
from pathlib import Path
from pprint import pprint
from typing import Any, Literal

from anthropic import Anthropic
from dotenv import load_dotenv
from inspect_ai import Task, eval, task
from inspect_ai.dataset import Dataset, Sample, example_dataset, hf_dataset, json_dataset
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.scorer import Score, Scorer, Target, answer, match, model_graded_fact, scorer
from inspect_ai.solver import (
    Choices,
    Generate,
    Solver,
    TaskState,
    chain,
    chain_of_thought,
    generate,
    self_critique,
    solver,
)
from openai import OpenAI

# Make sure exercises are in the path
chapter = "chapter3_llm_evals"
section = "part3_running_evals_with_inspect"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part3_running_evals_with_inspect.tests as tests

MAIN = __name__ == "__main__"

# %%
load_dotenv()

assert os.getenv("OPENROUTER_API_KEY") is not None, "You must set your OpenRouter API key - see instructions in dropdown"

openrouter_client = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")

# %%
import json

with open(exercises_dir / "part2_dataset_generation" / "power-seeking_300_qs.json") as f:
    dataset = json.load(f)

print(list(dataset[0].keys()))
for key, value in dataset[0].items():
    print(key, ':', value)

# %%
def record_to_sample(record: dict) -> Sample:
    """
    Converts a item ("record") from the dataset into a Sample object, mapping the fields of the
    record to the fields of the Sample object.

    Args:
        record : A dictionary from the json dataset containing our evaluation questions

    Returns:
        Sample : A Sample object containing the information in the record
    """
    print(record['answers'])
    return Sample(
        input=[ChatMessageSystem(content=record['system']), ChatMessageUser(content=record['user'])],
        target=record['answer_matching_behavior'],
        choices=[record['answers']],
        metadata={},
    )


# Edit these variables depending on what you saved yesterday!
evaluation_target = "power-seeking"
num_qs_saved = 300

json_dataset_path = str(exercises_dir / "part2_dataset_generation" / f"{evaluation_target}_{num_qs_saved}_qs.json")
my_dataset = json_dataset(json_dataset_path, record_to_sample)

# Pretty-print the data in the Samples object, so we can see its structure
pprint(my_dataset.samples[0].__dict__)

# %%
