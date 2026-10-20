# Shutdown Avoidance

## Getting Started

### Install dependencies

#### via nix+direnv

If you have [nix](https://nixos.org/) and [direnv](https://direnv.net/) installed, you can install the dependencies with just:

    direnv allow

#### via manual installation

Otherwise, you'll need a Python package manager. We recommend [uv](https://docs.astral.sh/uv/getting-started/installation/).

With `uv`, you can either prefix your commands with `uv run` (as in `uv run inspect eval shutdown.py`) or you can activate the virtual environment:

    uv sync
    source .venv/bin/activate

### Add API keys

You will need API keys for any LLM providers you want to use. (eg [OpenAI](https://platform.openai.com/api-keys)).

Make a new file called `.env` and paste your keys in there:

    OPENAI_API_KEY=<paste your key here>


## Running experiments

Run experiments with `inspect`:

    inspect eval shutdown.py --model openai/o3

You can view results with:

    inspect view

Learn more about inspect at https://inspect.aisi.org.uk
