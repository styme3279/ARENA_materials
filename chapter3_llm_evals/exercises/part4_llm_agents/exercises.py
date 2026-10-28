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

EVAL_MODEL = "openai/gpt-4o-mini"
os.environ["INSPECT_EVAL_MODEL"] = EVAL_MODEL
MAIN = __name__ == "__main__"

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
        # evaluate_expression()
        answers = [evaluate_expression(f"{self.num1} {operation} {self.num2}") for operation in self.operations]
        return answers

    @property
    def get_current_task(self) -> str:
        return f"{self.num1} {self.operations[self.current_task_number]} {self.num2}"

    def update_current_task(self) -> None:
        """
        Increments self.current_task_number by one (modulo the number of operations)
        """
        # raise NotImplementedError("You need to implement update_current_task")
        self.current_task_number = (self.current_task_number + 1) % len(self.operations)

    def get_current_instruction(self) -> ChatMessageUser:
        # raise NotImplementedError("You need to implement get_current_instruction")
        return ChatMessageUser(
            content=f"Calculate the following expression {self.get_current_task}. Give your answer in the format <ANSWER>NUMBER</ANSWER> where NUMBER is a numerical value formatted as a float."
        )




tests.test_arithmetic_task(ArithmeticTask)

arithmetic_task1 = ArithmeticTask(3, 5)
print(arithmetic_task1.get_current_task)
arithmetic_task1.update_current_task()
print(arithmetic_task1.get_current_task)
print(1)
print(arithmetic_task1.get_current_instruction())
# %%
@tool
def calculate():
    async def execute(expression: str) -> str:
        """
        This tool takes in an expression as an string and returns an string containing the evaluated answer.

        Args: 
        """
        print("hello")
        return "hi"
        # answer = evaluate_expression(expression)
    
        # return answer
    return calculate


tests.test_calculate_tool(calculate)

# %%
