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
import functools
import os
import re
import sys
import time
from pathlib import Path
from typing import Literal
from urllib.parse import unquote

import requests
import wikipedia
from bs4 import BeautifulSoup
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

# Wikipedia rate-limits bursts of API requests (e.g. 50 games starting at once) and answers with a
# non-JSON error page, which the `wikipedia` library surfaces as a JSONDecodeError. All of its requests
# go through `_wiki_request`, so retry there with backoff. The sleep blocks the event loop, which pauses
# every running game while we're rate-limited.
_wiki_request = getattr(wikipedia.wikipedia._wiki_request, "__wrapped__", wikipedia.wikipedia._wiki_request)


@functools.wraps(_wiki_request)
def _wiki_request_with_retries(params, retries: int = 7):
    for attempt in range(retries):
        try:
            return _wiki_request(params)
        except requests.exceptions.JSONDecodeError:
            if attempt == retries - 1:
                raise
            print(f"Wikipedia API returned an error page, retrying in {2**attempt}s")
            time.sleep(2**attempt)


wikipedia.wikipedia._wiki_request = _wiki_request_with_retries

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
@functools.cache
def article_links(title: str) -> tuple[str, ...]:
    """Links a reader can click in the article itself (text, lists, tables and infobox), not in navigation templates.

    Uses the rendered HTML, so links whose visible text differs from the title ("[[Pope Sixtus I|Sixtus I]]",
    "[[Vogue (magazine)|Vogue]]") count, and navboxes (which link e.g. every pope from every pope's page) don't.
    """
    html = wikipedia.wikipedia._wiki_request({"action": "parse", "page": title, "prop": "text", "redirects": 1})["parse"]["text"]["*"]
    soup = BeautifulSoup(html, "html.parser")
    for el in soup.select(".navbox, .sidebar, .reflist, .references, .hatnote, .metadata"):
        el.decompose()
    links = []
    for a in soup.select('a[href^="/wiki/"]:not(.new)'):  # .new = red link to a missing page
        target = unquote(a["href"].removeprefix("/wiki/")).split("#")[0].replace("_", " ")
        if target and ":" not in target:  # skip File:, Category:, Help: etc.
            links.append(target)
    return tuple(dict.fromkeys(links))


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
    @functools.cache  # games (and models) share page objects, so each page's content and links are fetched once
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
        Returns a list of permitted links for the current page: links in the article itself (text, lists, tables
        and infobox), but not in navigation boxes, sidebars or references.

        Returns:
            list[str]: The permitted links.
        """
        return [link for link in article_links(self.current_page.title) if link != self.current_page.title]

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
    """The agent's view of the game: current page, goal page and its summary, path so far, and the unvisited links it can move to."""
    visited = {title.lower() for title in game.page_history}
    links = sorted({link for link in game.get_permitted_links() if link.lower() not in visited}, key=str.lower)
    return (
        f"You are on page: {game.current_page.title}\n"
        f"Goal page: {game.goal_page.title}\n"
        f"About the goal page: {game.get_page_summary(game.goal_page)}\n"
        f"Your path so far: {' -> '.join(game.page_history)}\n"
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
WIKI_SYSTEM_PROMPT = """You are playing the Wikipedia game. Starting from one Wikipedia article, reach the goal article by following links between articles, using as few moves as possible.

Rules:
- Each turn you see the article you're on and the links you can follow from it. Call MovePageTool once with exactly one of those links.
- The links offered are the ones in the article itself: its text, lists, tables and infobox. Links in the navigation boxes at the bottom of articles don't count.
- Every turn counts towards your score, including failed moves and turns without a move.

Strategy:
- If the goal is among the links, move to it immediately.
- Otherwise, think about which articles would link to the goal: topics closely associated with it, such as its category or parent topic, related people, works, organisations, places, events and dates, and other articles with a similar or shared name. Plan a short route into that neighbourhood.
- From an unrelated starting point, one broad, well-connected article (a country, a major field, a large institution) can bridge you into the right area. Once you are close, prefer specific articles tied to the goal over general overviews.
- If you're in the right area but the goal isn't offered, try a different kind of related article rather than more of the same kind.
- Articles you've already visited are not offered again."""


@agent
def WikiAgent(tools: list[Tool], game: WikiGame, max_steps: int = 40, reasoning_effort: str | None = None):
    system_instruction = ChatMessageSystem(content=WIKI_SYSTEM_PROMPT)

    async def execute(state: AgentState) -> AgentState:
        # Replace the Sample's empty input: the agent only sees the current page, the goal and the links
        state.messages = [system_instruction, ChatMessageUser(content=page_observation(game))]
        for _ in range(max_steps):
            if game.check_win():
                break
            state.output = await get_model().generate(
                input=state.messages,
                tools=tools,
                tool_choice="auto",  # forced tool use ("any") is rejected by some models, e.g. Claude Sonnet 5.5
                config=GenerateConfig(parallel_tool_calls=False, reasoning_effort=reasoning_effort),
            )
            # One move per turn, even if the provider ignores parallel_tool_calls=False
            state.output.message.tool_calls = (state.output.message.tool_calls or [])[:1]
            state.messages.append(state.output.message)
            if not state.output.message.tool_calls:
                # A turn without a move still counts as a step
                state.messages.append(ChatMessageUser(content="Call MovePageTool with one of the links listed for your current page."))
                continue
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
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import Generate, TaskState, solver

MAX_STEPS = 15

# 7 extra pairs (shortest path checked against our link rules), then the 18 provided `wiki_pairs`,
# in rough order of difficulty. "Ear Clearing" no longer exists on Wikipedia, so it's replaced by
# the page "Ear clearing" now redirects to.
wiki_eval_pairs = [
    ("Pizza", "Mount Vesuvius"),  # 1 step
    ("Coffee", "Ottoman Empire"),  # 1 step
    ("Albert Einstein", "Nuclear weapon"),  # 2 steps, via Manhattan Project
    ("Chess", "Artificial intelligence"),  # 2 steps, via Computer chess
    ("Basketball", "Canada"),  # 2 steps, via James Naismith
    ("Elizabeth I", "United States"),  # 2 steps, via Virginia
    ("Python (programming language)", "Artificial intelligence"),  # 2 steps, via Common Lisp
    ("Muban", "Commercial Law"),
    ("County Seat", "Saint Pierre and Miquelon"),
    ("Government of the United Kingdom", "Correlation"),
    ("Software license", "Impressionism"),
    ("Ptolemy", "Flax"),
    ("Joinery", "Amethyst"),
    ("WebKit", "Financial Instrument"),
    ("Federal Assembly (Switzerland)", "Giacomo Puccini"),
    ("Human Geography", "Charlie Kaufman"),
    ("Written Chinese", "Amphipoda"),
    ("Cape Verde", "Annual Publication"),
    ("Penance", "Patent Cooperation Treaty"),
    ("Polish Air Force", "Vogue India"),
    ("Ionizing radiation", "Pope Sixtus I"),
    ("FIFA Women's World Cup", "Ludwig Maximilian University of Munich"),
    ("Spinal Column", "Mentor Graphics"),
    ("Joseph Beuys", "RNA world"),
    ("Middle ear equalization", "Newspeak"),
]

wiki_dataset = [
    Sample(id=i, input="", target=goal, metadata={"start": start, "goal": goal})
    for i, (start, goal) in enumerate(wiki_eval_pairs, start=1)
]


@solver
def wiki_game_solver(max_steps: int = MAX_STEPS):
    """Play one game per sample: a fresh WikiGame from the sample's metadata, then WikiAgent on it."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        game = WikiGame(state.metadata["start"], state.metadata["goal"])
        agent_solver = as_solver(WikiAgent(tools=[MovePageTool(game)], game=game, max_steps=max_steps))
        state = await agent_solver(state, generate)
        # Record the outcome from the game itself, so the scorer doesn't have to parse the transcript
        state.store.set("won", game.check_win())
        state.store.set("steps", sum(m.role == "assistant" for m in state.messages))
        state.store.set("path", game.page_history)
        return state

    return solve


@scorer(metrics=[mean()])
def wiki_steps_scorer(max_steps: int = MAX_STEPS):
    """1.0 for reaching the goal in 1 step, down to 1/max_steps on the last allowed step; 0 if never reached."""

    async def score(state: TaskState, target: Target) -> Score:
        won, steps, path = state.store.get("won", False), state.store.get("steps", 0), state.store.get("path", [])
        return Score(
            value=(max_steps - steps + 1) / max_steps if won else 0.0,
            answer=" -> ".join(path),
            explanation=f"{'reached' if won else 'did not reach'} {target.text} in {steps} steps",
            metadata={"won": won, "steps": steps},
        )

    return score


@task
def wiki_eval_task(max_steps: int = MAX_STEPS) -> Task:
    return Task(
        dataset=wiki_dataset,
        solver=wiki_game_solver(max_steps),
        scorer=wiki_steps_scorer(max_steps),
        message_limit=2 * max_steps + 10,  # backstop only: the agent stops itself after max_steps
    )


WIKI_MODELS = ["openrouter/anthropic/claude-sonnet-5.5"]

# Same task (pairs, solver, scorer, MAX_STEPS) for every model: inspect runs one eval per model, in parallel
wiki_logs = eval(wiki_eval_task(), model=WIKI_MODELS, fail_on_error=False)


def format_wiki_score(score) -> str:
    if score is None:
        return "ERROR"
    if not score.metadata["won"]:
        return "0.00 (not reached)"
    return f"{score.value:.2f} ({score.metadata['steps']} steps)"


model_names = [log.eval.model.split("/")[-1] for log in wiki_logs]
# One {sample id: Score} dict per model (None where the sample errored)
scores_by_id = [{s.id: (s.scores or {}).get("wiki_steps_scorer") for s in log.samples} for log in wiki_logs]

print(f"{'pair':<70}" + "".join(f"{name:<22}" for name in model_names))
for s in wiki_logs[0].samples:
    pair = f"{s.id:>2}. {s.metadata['start']} -> {s.metadata['goal']}"
    print(f"{pair:<70}" + "".join(f"{format_wiki_score(scores[s.id]):<22}" for scores in scores_by_id))
print(f"{'mean score':<70}" + "".join(
    f"{log.results.scores[0].metrics['mean'].value if log.results else float('nan'):<22.3f}" for log in wiki_logs
))
print(f"{'reached goal':<70}" + "".join(
    f"{sum(bool(sc and sc.metadata['won']) for sc in scores.values())}/{len(scores)}".ljust(22) for scores in scores_by_id
))
print(f"{'errors (not in mean)':<70}" + "".join(
    f"{sum(sc is None for sc in scores.values())}".ljust(22) for scores in scores_by_id
))

# %%
