# %%
import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

chapter = "chapter1_transformer_interp"
repo = "ARENA_materials"
branch = "main"

# Install dependencies
try:
    import transformer_lens
except:
    %pip install "openai>=1.56.2,<4" einops datasets jaxtyping "sae-lens>=6.51.3,<7.0.0" openai tabulate umap-learn hdbscan git+https://github.com/ARENA-education/eindex.git git+https://github.com/ARENA-education/CircuitsVis.git@arena-tl4.1#subdirectory=python git+https://github.com/ARENA-education/sae_vis.git@v0.3.7+arena.2 "transformer_lens>=3.9,<4"

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
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import circuitsvis as cv
import einops
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch as t
from datasets import load_dataset
from dotenv import load_dotenv
from IPython.display import HTML, display
from jaxtyping import Bool, Float
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

device = t.device("cuda" if t.cuda.is_available() else "cpu")
dtype = t.bfloat16

# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
section = "part31_linear_probes"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section

import part31_linear_probes.tests as tests
import part31_linear_probes.utils as utils

MAIN = __name__ == "__main__"
# %%
# Set up paths to the cloned repos
# Adjust these if your repos are in a different location
GOT_ROOT = exercises_dir / "geometry-of-truth"  # geometry-of-truth repo
DD_ROOT = exercises_dir / "deception-detection"  # deception-detection repo

assert GOT_ROOT.exists(), f"Please clone geometry-of-truth repo to {GOT_ROOT}"
assert DD_ROOT.exists(), f"Please clone deception-detection repo to {DD_ROOT}"

GOT_DATASETS = GOT_ROOT / "datasets"
DD_DATA = DD_ROOT / "data"
# %%
load_dotenv(dotenv_path=str(exercises_dir / ".env"))
HF_TOKEN = os.getenv("HF_TOKEN")
assert HF_TOKEN, "Please set HF_TOKEN in your chapter1_transformer_interp/exercises/.env file"
# %%
MODEL_NAME = "meta-llama/Llama-2-13b-hf"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=dtype,
    device_map="auto",
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

NUM_LAYERS = len(model.model.layers)
D_MODEL = model.config.hidden_size
# Layer choices from the geometry-of-truth repo config for llama-2-13b. The paper
# found truth representations are concentrated in early-to-mid layers, and identified
# these specific layers via patching experiments (Section 3, "group (b)").
PROBE_LAYER = 14
INTERVENE_LAYER = 8

print(f"Model: {MODEL_NAME}")
print(f"Layers: {NUM_LAYERS}, Hidden dim: {D_MODEL}")
print(f"Probe layer: {PROBE_LAYER}, Intervene layer: {INTERVENE_LAYER}")
# %%
DATASET_NAMES = ["cities", "sp_en_trans", "larger_than"]

datasets = {}
for name in DATASET_NAMES:
    df = pd.read_csv(GOT_DATASETS / f"{name}.csv")
    datasets[name] = df
    print(f"\n{name}: {len(df)} statements ({df['label'].sum()} true, {(1 - df['label']).sum():.0f} false)")
    display(df.head(4))
# %%
class _StopForward(Exception):
    pass


def extract_activations(
    statements: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layers: list[int],
    batch_size: int = 25,
) -> dict[int, Float[Tensor, "n_statements d_model"]]:
    """
    Extract last-token hidden state activations from specified layers for a list of statements.
    """
    layers = sorted(set(layers))
    deepest = layers[-1]
    all_acts = {layer: [] for layer in layers}
    state = {}  # holds last-real-token indices for the current batch

    def make_hook(layer: int):
        def hook(module, args, output):
            # Decoder layers return a tuple in older transformers versions, a tensor in newer ones
            hidden = output[0] if isinstance(output, tuple) else output  # [batch, seq, d_model]
            idx = state["last_idx"].to(hidden.device)
            rows = t.arange(hidden.shape[0], device=hidden.device)
            all_acts[layer].append(hidden[rows, idx].float().cpu())
            if layer == deepest:
                raise _StopForward  # no need to run the remaining layers
        return hook

    handles = [model.model.layers[l].register_forward_hook(make_hook(l)) for l in layers]
    try:
        for i in range(0, len(statements), batch_size):
            batch = statements[i : i + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(model.device)

            # Last non-padding position per row (works for left or right padding)
            mask = inputs["attention_mask"]
            state["last_idx"] = (mask * t.arange(mask.shape[1], device=mask.device)).argmax(dim=1)

            with t.no_grad():
                try:
                    model(**inputs, use_cache=False)
                except _StopForward:
                    pass
    finally:
        for h in handles:
            h.remove()  # always clean up, otherwise hooks leak into later cells

    return {layer: t.cat(all_acts[layer], dim=0) for layer in layers}


tests.test_extract_activations(extract_activations, model, tokenizer, PROBE_LAYER, D_MODEL)
# %%
# Extract activations at the probe layer for all datasets
activations = {}
labels_dict = {}
statements_dict = {}

for name in DATASET_NAMES:
    df = datasets[name]
    statements = df["statement"].tolist()
    labs = t.tensor(df["label"].values, dtype=t.float32)
    statements_dict[name] = statements

    acts = extract_activations(statements, model, tokenizer, [PROBE_LAYER])
    activations[name] = acts[PROBE_LAYER]
    labels_dict[name] = labs

# Show summary table
summary = pd.DataFrame(
    {
        "Dataset": DATASET_NAMES,
        "N statements": [len(datasets[n]) for n in DATASET_NAMES],
        "N true": [int(datasets[n]["label"].sum()) for n in DATASET_NAMES],
        "N false": [int((1 - datasets[n]["label"]).sum()) for n in DATASET_NAMES],
        "Act shape": [str(tuple(activations[n].shape)) for n in DATASET_NAMES],
        "Mean norm": [f"{activations[n].norm(dim=-1).mean():.1f}" for n in DATASET_NAMES],
    }
)
display(summary)
# %%
def get_pca_components(
    activations: Float[Tensor, "n d_model"],
    k: int = 2,
) -> Float[Tensor, "d_model k"]:
    """
    Compute the top-k principal components of the activation matrix.

    Args:
        activations: Activation matrix, shape [n_samples, d_model].
        k: Number of principal components to return.

    Returns:
        Matrix of top-k eigenvectors as columns, shape [d_model, k].
    """
    


tests.test_get_pca_components(get_pca_components, activations["cities"], D_MODEL)