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


def run_test(test_fn, *args):
    """
    Run one of the `tests.*` helpers in a fresh thread.

    Several tests call `asyncio.run(...)`, which raises
    `RuntimeError: asyncio.run() cannot be called from a running event loop`
    inside Jupyter / the VS Code interactive window (the kernel already owns a loop).
    A new thread has no running loop, so `asyncio.run` works there, and `.result()`
    re-raises any assertion failure so the test still fails loudly.
    """
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(test_fn, *args).result()

# Wikipedia's API rejects requests with the default `wikipedia` user-agent (see
# https://phabricator.wikimedia.org/T400119), so set our own before any
# `wikipedia.page(...)` call below.
wikipedia.set_user_agent("ARENA-AI-safety-course/3.4 (https://learn.arena.education)")

EVAL_MODEL = "openrouter/openai/gpt-4o-mini"
os.environ["INSPECT_EVAL_MODEL"] = EVAL_MODEL
MAIN = __name__ == "__main__"

# %%
load_dotenv()

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
        out_list = []
        for operation in self.operations:
            out_list.append(str(evaluate_expression(f"{self.num1} {operation} {self.num2}")))
            
        return out_list
        

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


tests.test_arithmetic_task(ArithmeticTask)

arithmetic_task1 = ArithmeticTask(3, 5)
print(arithmetic_task1.get_current_task)
arithmetic_task1.update_current_task()
print(arithmetic_task1.get_current_task)
print(arithmetic_task1.get_current_instruction())
# %%
@tool 
def tool_name(function_args):
    async def execute(tool_args : type) -> Tool:
        # tool logic
        return tool_output
    return execute

# %%
@tool 
def tool_name():
    def execute(sentence : str, n : int) -> str:
        """
        This tool appends a number to the end of the string.

        Args:
            sentence: this is the string to which you want to append a number.
            n: this is the number you want to append.

        Returns: 
            The sentence with the number appended.
        """
        return sentence + str(n)


# %%
@tool
def calculate():
    async def execute(expression: str) -> str:
        """
        This tool executes an expression to the end of the string.

        Args:
            expression: this is the string you want to be evaluated

        Returns: 
            The output of the expression
        """
        try:
            return str(evaluate_expression(expression))
        except Exception as e:
            return f"Error: {e}"
    return execute


run_test(tests.test_calculate_tool, calculate)

# %%
@agent
def arithmetic_agent(task: ArithmeticTask):
    async def execute(state: AgentState) -> AgentState:
        answer_list = ["wrong"] * len(task.operations)
        success = False
        while not success:
            state.messages.append(task.get_current_instruction())
            state.output = await get_model().generate(input=state.messages, tools=[calculate()], tool_choice="auto")
            state.messages.append(state.output.message)
            if state.output.message.tool_calls:
                messages, state.output = await execute_tools(state.messages, tools=[calculate()])
                state.messages.extend(messages)
            state.output = await get_model().generate(input=state.messages, tools=[calculate()], tool_choice="none")
            state.messages.append(state.output.message)
            try:
                if extract_answer(state.output.message.content) == task._generate_answers()[task.current_task_number]:
                    answer_list[task.current_task_number] = extract_answer(state.output.message.content)
                    task.update_current_task()

                else:
                    state.messages.append(ChatMessageUser(content="Incorrect answer. Try again."))
            except IndexError:
                state.messages.append(ChatMessageUser(content="Error: Could not extract answer"))
            if all(ans == task._generate_answers()[i] for i, ans in enumerate(answer_list)):
                success = True
        return state

    return execute


# %%
# @task
# def agent_task() -> str:
#     return Task(dataset=[Sample(input="", target="")], message_limit=40)

# eval(agent_task(), model=EVAL_MODEL, solver=as_solver(arithmetic_agent(task=ArithmeticTask(3, 5))))

# %%
# Retrieve a Wikipedia page from its title
page = wikipedia.page("Large language model")

# Access basic page information
print("Title:", page.title)
print("\nURL", page.url)
print(f"\nSummary (word count {len(page.summary.split())}):", page.summary)
print(
    f"\nContent (word count {len(page.content.split())}):",
    page.content[:1000],
    "......",
)
print(f"""\nLinks (link count {len(page.links)}): [{", ".join(page.links[:7])}, ......]""")

# %%
try:
    page = wikipedia.page("Python", auto_suggest=False)
except DisambiguationError as e:
    print(type(e), "\n\n", e)

# %%
try:
    page = wikipedia.page("Animalss", auto_suggest=False)
except Exception as e:
    print(type(e), "\n\n", e)

# %%
# Fixes PageError by allowing redirects
page = wikipedia.page("Animalss", redirect=True)
print(page.title)

# Fixes DisambiguationError by selecting the first option

try:
    page = wikipedia.page("Python", auto_suggest=False)
except DisambiguationError as e:
    page = wikipedia.page(e.options[0], auto_suggest=False)
print(page.title)

# %%
def get_page(title: str) -> WikipediaPage:
    """
    Get a Wikipedia page object given a title. If the title is ambiguous, choose the first option.
    If the title is not found, try to find a similar title.

    Args:
        title (str): The title of the Wikipedia page

    Returns:
        WikipediaPage: The Wikipedia page
    """
    try:
        return wikipedia.page(title, auto_suggest=False, redirect=True)
    except DisambiguationError as e:
        return wikipedia.page(e.options[0], auto_suggest=False, redirect=True)
    except PageError:
        return wikipedia.page(title, auto_suggest=True, redirect=True)

# %%
def get_permitted_links(current_page: WikipediaPage) -> list[str]:
    """
    Get "permitted" links (i.e. links that are in the content of the page) from a Wikipedia page.

    Args:
        current_page (WikipediaPage): The current Wikipedia page

    Returns:
        list[str]: A list of permitted links from current_page

    """
    all_links = current_page.links
    content_lower = current_page.content.lower()
    permitted_links = [link for link in all_links if link.lower() in content_lower]
    if current_page.title in permitted_links:
        permitted_links.remove(current_page.title)
    return permitted_links


tests.test_get_permitted_links(get_permitted_links)

# %%
class WikiGame:
    def __init__(
        self,
        starting_page: str,
        goal_page: str,
    ):
        """
        This task simulates the Wikipedia game, where the agent starts on one Wikipedia page and
        attempts to navigate to a goal page using only links found in the main content of Wikipedia
        pages.

        Args:
            starting_page (str): The page the agent starts on.
            goal_page (str): The page the agent is trying to reach.

        Attributes:
            page_history (list[str]): The history of pages visited by the agent.
            starting_page (WikipediaPage): The starting page of the game.
            goal_page (WikipediaPage): The goal page of the game.
            current_page (WikipediaPage): The current page the agent is on.

        """
        self.page_history: list[str] = [starting_page]
        self.starting_page: WikipediaPage = self.get_page(starting_page)
        self.goal_page: WikipediaPage = self.get_page(goal_page)
        self.current_page: WikipediaPage = self.starting_page

    # ========================= Helper Functions (given) =========================

    # Get page and page summary
    @staticmethod
    def get_page(title: str) -> WikipediaPage:
        """
        Get a Wikipedia page object given a title. If the title is ambiguous, choose the first
        option. If the title is not found, try to find a similar title.

        Args:
            title (str): The title of the Wikipedia page

        Returns:
            WikipediaPage: The Wikipedia page
        """
        try:
            return wikipedia.page(title, auto_suggest=False, redirect=True)
        except DisambiguationError as e:
            return wikipedia.page(e.options[0], auto_suggest=False, redirect=True)
        except PageError:
            return wikipedia.page(title, auto_suggest=True, redirect=True)

    def get_page_summary(self, page: WikipediaPage | None = None) -> str:
        """
        Get summary of a wikipedia page, to the last full stop within the first 500 characters.
        This can be used to give a brief overview of a page to the agent.

        Args:
            page (WikipediaPage): The Wikipedia page object.

        Returns:
            str: The summary of the Wikipedia page.
        """
        page = page if page else self.goal_page
        summary = page.content[:500]
        last_period_index = summary.rfind(".")
        return summary[: last_period_index + 1] if last_period_index != -1 else summary

    # Get and check permitted links
    def get_permitted_links(self) -> list[str]:
        """
        Returns a list of permitted links (i.e. links in the main page content) for the current page.

        Returns:
            list[str]: The permitted links.
        """
        all_links = self.current_page.links
        content_lower = self.current_page.content.lower()
        permitted_links = [link for link in all_links if link.lower() in content_lower]
        if self.current_page.title in permitted_links:
            permitted_links.remove(self.current_page.title)
        return permitted_links

    def is_permitted_link(self, link: str) -> bool:
        """
        Returns True if the link is in the permitted links for the current page, False otherwise.

        Args:
            link (str): The link to check.

        Returns:
            bool: True if the link is permitted, False otherwise
        """
        return link.lower() in (x.lower() for x in self.get_permitted_links())

    # ========================= Task State Management (given) =========================

    def check_win(self) -> bool:
        return self.current_page == self.goal_page

# %%
