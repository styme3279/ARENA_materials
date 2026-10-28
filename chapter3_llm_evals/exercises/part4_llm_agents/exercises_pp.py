import os
import sys
import warnings
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

chapter = "chapter3_llm_evals"
repo = "ARENA_materials"
branch = "main"

# Install dependencies
try:
    import inspect_ai
except:
    %pip install "openai>=1.56.2,<4" anthropic inspect_ai tabulate wikipedia jaxtyping

# Get root directory, handling 3 different cases: (1) Colab, (2) notebook not in ARENA repo, (3) notebook in ARENA repo
root = (
    "/content"
    if IN_COLAB
    else "/root"
    if repo not in os.getcwd()
    else str(next(p for p in Path.cwd().parents if p.name == repo))
)

if Path(root).exists() and not Path(f"{root}/{chapter}").exists():
    if not IN_COLAB:
        !sudo apt-get install unzip
        %pip install jupyter ipython --upgrade

    if not os.path.exists(f"{root}/{chapter}"):
        !wget -P {root} https://github.com/ARENA-education/ARENA_materials/archive/refs/heads/{branch}.zip
        !unzip {root}/{branch}.zip '{repo}-{branch}/{chapter}/exercises/*' -d {root}
        !mv {root}/{repo}-{branch}/{chapter} {root}/{chapter}
        !rm {root}/{branch}.zip
        !rmdir {root}/{repo}-{branch}

if IN_COLAB:
    from google.colab import userdata

    try:
        os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
    except:
        warnings.warn(
            "You don't have an OPENAI_API_KEY variable set in the secrets tab of your google colab. You have to set one, or any calls to APIs won't work."
        )


if f"{root}/{chapter}/exercises" not in sys.path:
    sys.path.append(f"{root}/{chapter}/exercises")

os.chdir(f"{root}/{chapter}/exercises")
# %%
import os
import re
import sys
from pathlib import Path
from typing import Literal

import wikipedia
from dotenv import load_dotenv
from inspect_ai import Task, eval, task
from inspect_ai.agent import Agent, AgentState, agent, as_solver
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    execute_tools,
    get_model,
)
from inspect_ai.tool import Tool, tool
from openai import OpenAI
from wikipedia import DisambiguationError, PageError, WikipediaPage

# Make sure exercises are in the path
chapter = "chapter3_llm_evals"
section = "part4_llm_agents"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part4_llm_agents.tests as tests
from part4_llm_agents.utils import evaluate_expression, execute_tools, extract_answer

# Wikipedia's API rejects requests with the default `wikipedia` user-agent (see
# https://phabricator.wikimedia.org/T400119), so set our own before any
# `wikipedia.page(...)` call below.
wikipedia.set_user_agent("ARENA-AI-safety-course/3.4 (https://learn.arena.education)")

EVAL_MODEL = "openrouter/openai/gpt-4o-mini"
os.environ["INSPECT_EVAL_MODEL"] = EVAL_MODEL
MAIN = __name__ == "__main__"
# %%
assert os.getenv("OPENROUTER_API_KEY") is not None, "You must set your OpenRouter API key - see instructions in dropdown"

# OPENROUTER_API_KEY

openai_client = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
# %%
class ArithmeticTask:
    def __init__(self, num1: int | float, num2: int | float, operations: list[str] | None = None):
        self.num1 = num1
        self.num2 = num2
        self.operations = operations if operations else ["+", "-", "*", "/", "**", "//", "%"]
        self.current_task_number = 0

    def _generate_answers(self) -> list[str]:
        """
        Generates a list of the correct answers for all the possible tasks

        Returns:
            list[str]: A list of the correct answers for all the possible tasks
        """
        answers = []
        for op in self.operations:
            try:
                result = evaluate_expression(f"{self.num1} {op} {self.num2}")
                answers.append(str(result))
            except Exception as e:
                answers.append(f"Error: {str(e)}")
        return answers

    @property
    def get_current_task(self) -> str:
        return f"{self.num1} {self.operations[self.current_task_number]} {self.num2}"

    def update_current_task(self) -> None:
        """
        Increments self.current_task_number by one (modulo the number of operations)
        """
        self.current_task_number = (self.current_task_number + 1) % len(self.operations)

    def get_current_instruction(self) -> ChatMessageUser:
        return ChatMessageUser(
            content=f"Calculate the following expression {self.get_current_task}. Give your answer in the format <ANSWER>NUMBER</ANSWER> where NUMBER is a numerical value formatted as a float."
        )


arithmetic_task1 = ArithmeticTask(3, 5)
print(arithmetic_task1.get_current_task)
arithmetic_task1.update_current_task()
print(arithmetic_task1.get_current_task)
print(arithmetic_task1.get_current_instruction())
# %%
import importlib; importlib.reload(tests)
@tool
def calculate():
    async def execute(expression: str) -> str:
        """
        This tool calculates the result of an arithmetical expression.

        Args:
            expression: this is the arithmetical expression you want to calculate.
        
        Returns:
            str: the result of the arithmetical expression
        """
        return str(evaluate_expression(expression))

    return execute


tests.test_calculate_tool(calculate)
# %%
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import TaskState


@scorer(metrics=[mean()])
def arithmetic_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        expected = target.target  # list: one answer per operation, in order
        solved = 0
        for msg in state.messages:
            if solved == len(expected):
                break
            if msg.role == "assistant" and extract_answer(msg.text) == expected[solved]:
                solved += 1
        return Score(
            value=solved / len(expected),
            answer=str(solved),
            explanation=f"solved {solved}/{len(expected)} in order",
        )

    return score
# %%
arith = ArithmeticTask(3, 5)

@task
def agent_task() -> Task:
    return Task(
        dataset=[Sample(input="", target=arith._generate_answers())],
        scorer=arithmetic_scorer(),
        message_limit=40,
    )

# %%
eval(agent_task(), solver=as_solver(arithmetic_agent(task=arith)))
# %%
arith = ArithmeticTask(3, 5)

@task
def agent_task() -> Task:
    return Task(
        dataset=[Sample(
            input="",
            target=arith._generate_answers(),
            metadata={"questions": [f"{arith.num1} {op} {arith.num2}" for op in arith.operations]},
        )],
        scorer=arithmetic_scorer(),
        message_limit=60,
    )

@scorer(metrics=[mean()])
def arithmetic_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        expected = target.target
        questions = state.metadata["questions"]
        tries = [[] for _ in expected]  # model's answers, per question
        solved = 0
        for msg in state.messages:
            if solved == len(expected):
                break
            if msg.role != "assistant" or msg.tool_calls:
                continue  # only final answers, not tool-call turns
            answer = extract_answer(msg.text)
            tries[solved].append(answer)
            if answer == expected[solved]:
                solved += 1

        lines = []
        for i, (q, exp) in enumerate(zip(questions, expected)):
            status = "✓" if i < solved else "✗" if tries[i] else "not reached"
            lines.append(f"- {status} `{q}` expected `{exp}`, model said {tries[i]}")
        return Score(
            value=solved / len(expected),
            answer=str(solved),
            explanation="\n".join(lines),
        )

    return score
# %%