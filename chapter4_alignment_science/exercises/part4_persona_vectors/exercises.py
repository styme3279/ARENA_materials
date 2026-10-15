# %%
import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

chapter = "chapter4_alignment_science"
repo = "ARENA_materials"
branch = "main"

# Install dependencies
try:
    import transformer_lens
except:
    %pip install "transformer_lens>=3.9,<4" einops jaxtyping "openai>=1.56.2,<4"

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


if f"{root}/{chapter}/exercises" not in sys.path:
    sys.path.append(f"{root}/{chapter}/exercises")

os.chdir(f"{root}/{chapter}/exercises")
# %%
import gc
import json
import os
import re
import sys
import textwrap
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import torch as t
import torch.nn.functional as F
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, login, snapshot_download
from IPython.display import HTML, display
from jaxtyping import Float
from openai import OpenAI
from sklearn.decomposition import PCA
from torch import Tensor
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

t.set_grad_enabled(False)

# Make sure exercises are in the path
chapter = "chapter4_alignment_science"
section = "part4_persona_vectors"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section

import part4_persona_vectors.tests as tests
import part4_persona_vectors.utils as utils

warnings.filterwarnings("ignore")

device = t.device("cuda" if t.cuda.is_available() else "cpu")
dtype = t.bfloat16

MAIN = __name__ == "__main__"


def print_with_wrap(s: str, width: int = 80):
    """Print text with line wrapping, preserving newlines."""
    out = []
    for line in s.splitlines(keepends=False):
        out.append(textwrap.fill(line, width=width) if line.strip() else line)
    print("\n".join(out))
# %%
assistant_axis_path = exercises_dir / "assistant-axis"
assert assistant_axis_path.exists(), "Please clone the assistant-axis repo (see instructions above)"

transcript_dir = assistant_axis_path / "transcripts"
case_study_files = sorted(transcript_dir.glob("case_studies/**/*.json"))
drift_files = sorted(transcript_dir.glob("persona_drift/*.json"))
print(f"Found {len(case_study_files)} case study transcripts, {len(drift_files)} persona drift transcripts")

# Show available transcripts
for f in case_study_files:
    data = json.loads(f.read_text())
    print(f"  Case study: {f.parent.name}/{f.stem} ({data.get('turns', '?')} turns, model={data.get('model', '?')})")
for f in drift_files:
    data = json.loads(f.read_text())
    print(f"  Persona drift: {f.stem} ({data.get('turns', '?')} turns, model={data.get('model', '?')})")
# %%
env_path = exercises_dir / ".env"
assert env_path.exists(), "Please create a .env file with your API keys"

load_dotenv(dotenv_path=str(env_path))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
assert OPENROUTER_API_KEY, "Please set OPENROUTER_API_KEY in your .env file"

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)
# %%
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)
# %%
