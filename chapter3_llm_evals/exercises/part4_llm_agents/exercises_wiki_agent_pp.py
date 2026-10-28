# %%
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
    GenerateConfig,
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
def page_observation(game: WikiGame) -> str:
    """The agent's view of the game: current page, goal page, and the links it can move to."""
    links = sorted(set(game.get_permitted_links()), key=str.lower)
    return (
        f"You are on page: {game.current_page.title}\n"
        f"Goal page: {game.goal_page.title}\n"
        f"Links you can move to ({len(links)}):\n" + "\n".join(links)
    )


@tool
def GetContentTool(game: WikiGame) -> Tool:
    async def execute() -> str:
        """
        Get all the content for the wikipedia page you are currently on. Anything which corresponds to a link is wrapped in <link></link> tags.

        Args:
            None

        Returns:
            str: The content of the page with any accessible links wrapped in <link></link> tags
        """
        content = game.current_page.content
        permitted_links = get_permitted_links(game.current_page)
        for word in sorted(permitted_links, key=len, reverse=True):
            content = re.sub(
                r"""(\s|[,.)!?;:'"])(""" + re.escape(word) + r""")(\s|[,.)!?;:'"s])""",
                r"\1<link>\2</link>\3",
                content,
                count=1,
                flags=re.IGNORECASE,
            )
        return content

    return execute


@tool
def MovePageTool(game: WikiGame) -> Tool:
    async def execute(page: str) -> str:
        """
        Move to a new wikipedia page by following one of the links listed for the current page. Modifies the game state in place.

        Args:
            page: The title of the page you want to move to. This must be one of the links listed for the current page, or the move will fail.

        Returns:
            str: Whether the move was successful, followed by the new page and the links you can move to from it
        """
        page_no_underscore = page.replace("_", " ")
        if game.is_permitted_link(page):
            new_page = game.get_page(page)
            game.current_page = new_page
            game.page_history.append(new_page.title)
            return "Move successful.\n\n" + page_observation(game)
        elif game.is_permitted_link(page_no_underscore):
            new_page = game.get_page(page_no_underscore)
            game.current_page = new_page
            game.page_history.append(new_page.title)
            return "Move successful.\n\n" + page_observation(game)
        else:
            return f"Move failed: {page!r} is not a link on {game.current_page.title}. Pick one of the links listed for this page."

    return execute
# %%
@agent
def WikiAgent(tools: list[Tool], game: WikiGame):
    system_instruction = ChatMessageSystem(
        content="You are a wikipedia-racing AI. Your aim is to reach the goal page by moving from page to page along links. "
        "Each turn, call MovePageTool once with one of the links listed for your current page."
    )

    async def execute(state: AgentState) -> AgentState:
        # Replace the Sample's empty input: the agent only sees the current page, the goal and the links
        state.messages = [system_instruction, ChatMessageUser(content=page_observation(game))]
        while not game.check_win():
            state.output = await get_model().generate(
                input=state.messages,
                tools=tools,
                tool_choice="any",
                config=GenerateConfig(parallel_tool_calls=False),
            )
            # One move per turn, even if the provider ignores parallel_tool_calls=False
            state.output.message.tool_calls = (state.output.message.tool_calls or [])[:1]
            state.messages.append(state.output.message)
            messages, _ = await execute_tools(messages=state.messages, tools=tools)
            state.messages.extend(messages)
        return state

    return execute
# %%
game = WikiGame("Python (programming language)", "Artificial intelligence")
tool_list = [MovePageTool(game)]


@task
def wiki_task() -> Task:
    return Task(dataset=[Sample(input="", target="")], message_limit=20)


eval(
    solver=as_solver(WikiAgent(tools=tool_list, game=game)),
    tasks=wiki_task(),
)
# %%
game_1 = WikiGame("Elizabeth I", "United States")
tool_list = [MovePageTool(game_1)]


@task
def wiki_task() -> Task:
    return Task(dataset=[Sample(input="", target="")], message_limit=80)


eval(
    solver=as_solver(WikiAgent(tools=tool_list, game=game_1)),
    tasks=wiki_task(),
)
game_2 = WikiGame("County Seat", "Saint Pierre and Miquelon")
tool_list = [MovePageTool(game_2)]


@task
def wiki_task() -> Task:
    return Task(dataset=[Sample(input="", target="")], message_limit=80)


eval(
    solver=as_solver(WikiAgent(tools=tool_list, game=game_2)),
    tasks=wiki_task(),
)
# %%