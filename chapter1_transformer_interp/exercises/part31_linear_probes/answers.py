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
def extract_activations(
    statements: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layers: list[int],
    batch_size: int = 25,
) -> dict[int, Float[Tensor, "n_statements d_model"]]:
    """
    Extract last-token hidden state activations from specified layers for a list of statements.

    Args:
        statements: List of text statements to process.
        model: A HuggingFace causal language model.
        tokenizer: The corresponding tokenizer.
        layers: List of layer indices (0-indexed) to extract activations from.
        batch_size: Number of statements to process at once.

    Returns:
        Dictionary mapping layer index to tensor of activations, shape [n_statements, d_model].
    """
    all_acts = {layer: [] for layer in layers}
    for i in range(0, len(statements), batch_size):
        batch = statements[i : i + batch_size]
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with t.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
        last_token_idx = tokens['attention_mask'].sum(dim=1) - 1
        
        for layer in layers:
            hidden = outputs.hidden_states[layer+1]
            batch_indices = t.arange(len(last_token_idx), device=hidden.device)
            acts = hidden[batch_indices, last_token_idx]
            all_acts[layer].append(acts.cpu().float())
    
    return {layer: t.cat(acts_list, dim=0) for layer, acts_list in all_acts.items()}

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
    # mean center
    # print(f"activations: {activations.shape}")
    mean_acts = activations.mean(dim=0, keepdim=True)
    # print(f"mean_acts: {mean_acts.shape}")
    centered_activations = (activations - mean_acts).T
    # print(f"centered_activations: {centered_activations.shape}")

    # covariance
    covariance = t.cov(centered_activations)
    # print(f"covariacne: {covariance.shape}")

    # eigendec
    eigenval, eigenvec = t.linalg.eigh(covariance)

    return eigenvec[:, -k:].flip(dims=(-1,))


tests.test_get_pca_components(get_pca_components, activations["cities"], D_MODEL)


# %%
fig = make_subplots(rows=1, cols=3, subplot_titles=DATASET_NAMES)

for i, name in enumerate(DATASET_NAMES):
    acts = activations[name]
    label_text = labels_dict[name]
    prompts = statements_dict[name]
    pcs = get_pca_components(acts, k=2)
    X_centered = acts - acts.mean(dim=0)
    projected = (X_centered @ pcs).numpy()

    # Compute variance explained
    total_var = X_centered.var(dim=0).sum().item()
    pc_var = t.tensor(projected).var(dim=0)
    pct_explained = (pc_var / total_var * 100).tolist()

    colors = ["blue" if l == 1 else "red" for l in label_text.tolist()]
    fig.add_trace(
        go.Scatter(
            x=projected[:, 0],
            y=projected[:, 1],
            mode="markers",
            marker=dict(color=colors, size=3, opacity=0.5),
            name=name,
            showlegend=False,
            hovertext=prompts,
            customdata=list(zip(prompts, label_text)),
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                "%{customdata[0]}<br>"
                "PC1: %{x:.2f}<br>"
                "PC2: %{y:.2f}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=i + 1,
    )
    fig.update_xaxes(title_text=f"PC1 ({pct_explained[0]:.1f}%)", row=1, col=i + 1)
    fig.update_yaxes(title_text=f"PC2 ({pct_explained[1]:.1f}%)", row=1, col=i + 1)

# Add a legend manually
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color="blue", size=8), name="True"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color="red", size=8), name="False"))

fig.update_layout(
    title="PCA of Truth Representations (Layer 14, Last Token)",
    height=400,
    width=1200,
)
fig.show()

# %%
def layer_sweep_accuracy(
    statements: list[str],
    labels: Float[Tensor, " n"],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layers: list[int],
    train_frac: float = 0.8,
    batch_size: int = 25,
) -> dict[str, list[float]]:
    """
    For each layer, train a difference-of-means classifier and compute train/test accuracy.

    Args:
        statements: List of statements.
        labels: Binary labels (1=true, 0=false).
        model: The language model.
        tokenizer: The tokenizer.
        layers: List of layer indices to sweep over.
        train_frac: Fraction of data for training.
        batch_size: Batch size for activation extraction.

    Returns:
        Dict with keys "train_acc" and "test_acc", each a list of accuracies per layer.
    """
    n_statements = len(statements)
    print(f"num_statements: {n_statements}")
    print(f"labels: {labels}")

    n_train = int(n_statements * train_frac)
    perm = t.randperm(n_statements)
    print(f"perm[:n_train]: {perm[:n_train]}")
    # train_statements = statements[perm[:n_train]]
    train_statements = [statements[i] for i in perm[:n_train]]
    test_statements = [statements[i] for i in perm[n_train:]]

    train_labels = [labels[i] for i in perm[:n_train]]
    test_labels = [labels[i] for i in perm[n_train:]]

    # test_statements = statements[perm[n_train:]]

    # Extract activations
    train_acts = extract_activations(train_statements, model, tokenizer, layers, batch_size)
    test_acts = extract_activations(test_statements, model, tokenizer, layers, batch_size)
    print(f'train_acts: {train_acts}')

    out = {"train_acc": [], "test_acc": []}

    for layer in layers:
        train_act_l = train_acts[layer] # b d_model
        test_act_l = test_acts[layer] # b d_model
        print(f"train_act_l shape: {train_act_l.shape}")
        
        indices_true = t.where(t.as_tensor(train_labels)==1)[0]
        indices_false = t.where(t.as_tensor(train_labels)==0)[0]
        print(f"indices_true: {indices_true}")
        train_mean_true = train_act_l[indices_true].mean(dim=0)
        train_mean_false = train_act_l[indices_false].mean(dim=0)

        train_direction = train_mean_true - train_mean_false
        print(f"train_direction: {train_direction.shape}")
        print(f"train_act_l: {train_act_l.shape}")

        center = (train_mean_true + train_mean_false) / 2

        dot_product = einops.einsum(train_act_l - center, train_direction, "b d_model, d_model -> b")
        predictions = [0 if dot < 0 else 1 for dot in dot_product ]

        train_acc = 0
        for i in range(len(predictions)):
            if predictions[i] == train_labels[i].item():
                train_acc += 1

        print(f"train_acc: {train_acc}")

        dot_product = einops.einsum(test_act_l - center, train_direction, "b d_model, d_model -> b")
        predictions = [0 if dot < 0 else 1 for dot in dot_product ]

        test_acc = 0
        for i in range(len(predictions)):
            if predictions[i] == test_labels[i].item():
                test_acc += 1
        train_acc /= train_act_l.shape[0]
        test_acc /= test_act_l.shape[0]
        out['train_acc'].append(train_acc)
        out['test_acc'].append(test_acc)


    return out


t.manual_seed(42)
all_layers = list(range(NUM_LAYERS))
cities_statements = datasets["cities"]["statement"].tolist()
cities_labels = t.tensor(datasets["cities"]["label"].values, dtype=t.float32)

sweep_results = layer_sweep_accuracy(cities_statements, cities_labels, model, tokenizer, all_layers)

# Print results as a table
sweep_df = pd.DataFrame(
    {
        "Layer": all_layers,
        "Train Acc": [f"{a:.3f}" for a in sweep_results["train_acc"]],
        "Test Acc": [f"{a:.3f}" for a in sweep_results["test_acc"]],
    }
)
display(sweep_df)

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=all_layers, y=sweep_results["train_acc"], mode="lines+markers", name="Train"))
fig.add_trace(go.Scatter(x=all_layers, y=sweep_results["test_acc"], mode="lines+markers", name="Test"))
fig.add_vline(x=PROBE_LAYER, line_dash="dash", line_color="gray", annotation_text=f"Probe layer ({PROBE_LAYER})")
fig.update_layout(
    title="Layer Sweep: Difference-of-Means Accuracy on Cities Dataset",
    xaxis_title="Layer",
    yaxis_title="Accuracy",
    yaxis_range=[0.4, 1.05],
    height=400,
    width=800,
)
fig.show()

best_layer = all_layers[int(np.argmax(sweep_results["test_acc"]))]
print(f"\nBest layer by test accuracy: {best_layer} ({max(sweep_results['test_acc']):.3f})")
print(f"Configured probe layer: {PROBE_LAYER} ({sweep_results['test_acc'][PROBE_LAYER]:.3f})")

# %%
# Create train/test splits for all datasets
t.manual_seed(42)
train_acts, test_acts = {}, {}
train_labels, test_labels = {}, {}

for name in DATASET_NAMES:
    acts = activations[name]
    labs = labels_dict[name]
    n = len(acts)
    perm = t.randperm(n)
    n_train = int(0.8 * n)

    train_acts[name] = acts[perm[:n_train]]
    test_acts[name] = acts[perm[n_train:]]
    train_labels[name] = labs[perm[:n_train]]
    test_labels[name] = labs[perm[n_train:]]

    print(f"{name}: train={n_train}, test={n - n_train}")

# %%
class MMProbe(t.nn.Module):
    def __init__(
        self,
        direction: Float[Tensor, " d_model"],
        covariance: Float[Tensor, "d_model d_model"] | None = None,
        atol: float = 1e-3,
        bias: Float[Tensor, " d_model"] | None = None,
    ):
        super().__init__()
        self.direction = t.nn.Parameter(direction, requires_grad=False)
        if bias is not None:
            self.bias = t.nn.Parameter(bias, requires_grad=False)
        else:
            self.bias = None
        if covariance is not None:
            self.inv = t.nn.Parameter(t.linalg.pinv(covariance, hermitian=True, atol=atol), requires_grad=False)
        else:
            self.inv = None

    def forward(self, x: Float[Tensor, "n d_model"], iid: bool = False) -> Float[Tensor, " n"]:
        if self.bias is not None:
            x = x - self.bias
        if iid and self.inv is not None:
            return t.sigmoid(x @ self.inv @ self.direction)
        else:
            return t.sigmoid(x @ self.direction)

    def pred(self, x: Float[Tensor, "n d_model"], iid: bool = False) -> Float[Tensor, " n"]:
        return self(x, iid=iid).round()

    @staticmethod
    def from_data(
        acts: Float[Tensor, "n d_model"],
        labels: Float[Tensor, " n"],
        device: str = "cpu",
        bias: bool = True,
    ) -> "MMProbe":
        acts, labels = acts.to(device), labels.to(device)
        pos_acts = acts[labels == 1]
        neg_acts = acts[labels == 0]
        pos_mean = pos_acts.mean(0)
        neg_mean = neg_acts.mean(0)
        direction = pos_mean - neg_mean

        centered = t.cat([pos_acts - pos_mean, neg_acts - neg_mean], dim=0)
        covariance = centered.t() @ centered / acts.shape[0]

        mu_mean = 0.5 * (pos_mean + neg_mean) if bias else None
        return MMProbe(direction, covariance=covariance, bias=mu_mean).to(device)
        


mm_probe = MMProbe.from_data(train_acts["cities"], train_labels["cities"])

# Train accuracy
train_preds = mm_probe.pred(train_acts["cities"])
train_acc = (train_preds == train_labels["cities"]).float().mean().item()

# Test accuracy
test_preds = mm_probe.pred(test_acts["cities"])
test_acc = (test_preds == test_labels["cities"]).float().mean().item()
assert test_acc > 0.9, "Expected at least 90% accuracy"

print("MMProbe on cities:")
print(f"  Train accuracy: {train_acc:.3f}")
print(f"  Test accuracy:  {test_acc:.3f}")
print(f"  Direction norm: {mm_probe.direction.norm().item():.3f}")
print(f"  Direction (first 5): {mm_probe.direction[:5].tolist()}")

# %%

class LRProbe(t.nn.Module):
    def __init__(self, d_in: int, scaler_mean: Tensor | None = None, scaler_scale: Tensor | None = None):
        super().__init__()
        self.net = t.nn.Sequential(t.nn.Linear(d_in, 1, bias=False), t.nn.Sigmoid())
        self.register_buffer("scaler_mean", scaler_mean)
        self.register_buffer("scaler_scale", scaler_scale)

    def _normalize(self, x: Float[Tensor, "n d_model"]) -> Float[Tensor, "n d_model"]:
        """Apply StandardScaler normalization if scaler parameters are available."""
        if self.scaler_mean is not None and self.scaler_scale is not None:
            return (x - self.scaler_mean) / self.scaler_scale
        return x

    def forward(self, x: Float[Tensor, "n d_model"]) -> Float[Tensor, " n"]:
        return self.net(self._normalize(x)).squeeze(-1)

    def pred(self, x: Float[Tensor, "n d_model"]) -> Float[Tensor, " n"]:
        return self(x).round()

    @property
    def direction(self) -> Float[Tensor, " d_model"]:
        return self.net[0].weight.data[0]

    @staticmethod
    def from_data(
        acts: Float[Tensor, "n d_model"],
        labels: Float[Tensor, " n"],
        C: float = 0.1,
        device: str = "cpu",
    ) -> "LRProbe":
        """
        Train an LR probe using sklearn's LogisticRegression with StandardScaler normalization.

        Args:
            acts: Activation matrix [n_samples, d_model].
            labels: Binary labels (1=true, 0=false).
            C: Inverse regularization strength (lower = stronger regularization).
                Default 0.1 (reg_coeff=10) matches the deception-detection paper's cfg.yaml.
                The repo class default is reg_coeff=1000 (C=0.001), which is stronger.
            device: Device to place the resulting probe on.
        """
        X = acts.cpu().float().numpy()
        y = labels.cpu().float().numpy()

        # Standardize features (zero mean, unit variance) before fitting, as in the paper
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # fit_intercept=False: the paper fits on normalized data so the intercept is redundant
        lr_model = LogisticRegression(C=C, random_state=42, fit_intercept=False, max_iter=1000)
        lr_model.fit(X_scaled, y)

        # Build probe with scaler parameters baked in
        scaler_mean = t.tensor(scaler.mean_, dtype=t.float32)
        scaler_scale = t.tensor(scaler.scale_, dtype=t.float32)
        probe = LRProbe(acts.shape[-1], scaler_mean=scaler_mean, scaler_scale=scaler_scale).to(device)
        probe.net[0].weight.data[0] = t.tensor(lr_model.coef_[0], dtype=t.float32).to(device)

        return probe

lr_probe = LRProbe.from_data(train_acts["cities"], train_labels["cities"], device="cpu")

# Train accuracy
train_preds = lr_probe.pred(train_acts["cities"])
train_acc = (train_preds == train_labels["cities"]).float().mean().item()

# Test accuracy
test_preds = lr_probe.pred(test_acts["cities"])
test_acc = (test_preds == test_labels["cities"]).float().mean().item()

print("LRProbe on cities:")
print(f"  Train accuracy: {train_acc:.3f}")
print(f"  Test accuracy:  {test_acc:.3f}")
print(f"  Direction norm: {lr_probe.direction.norm().item():.3f}")
assert test_acc >= 0.90, f"Test accuracy too low: {test_acc:.3f} (expected >= 0.90)"

# Compare directions
mm_dir = mm_probe.direction / mm_probe.direction.norm()
lr_dir = lr_probe.direction / lr_probe.direction.norm()
cos_sim = (mm_dir @ lr_dir).item()
print(f"\nCosine similarity between MM and LR directions: {cos_sim:.4f}")

# Compare both probes across all 3 datasets
results_rows = []
for name in DATASET_NAMES:
    mm_p = MMProbe.from_data(train_acts[name], train_labels[name])
    lr_p = LRProbe.from_data(train_acts[name], train_labels[name])

    mm_test_acc = (mm_p.pred(test_acts[name]) == test_labels[name]).float().mean().item()
    lr_test_acc = (lr_p.pred(test_acts[name]) == test_labels[name]).float().mean().item()
    results_rows.append({"Dataset": name, "MM Test Acc": f"{mm_test_acc:.3f}", "LR Test Acc": f"{lr_test_acc:.3f}"})

results_df = pd.DataFrame(results_rows)
print("\nProbe accuracy comparison across datasets:")
display(results_df)

# Bar chart
fig = go.Figure()
fig.add_trace(go.Bar(name="MMProbe", x=DATASET_NAMES, y=[float(r["MM Test Acc"]) for r in results_rows]))
fig.add_trace(go.Bar(name="LRProbe", x=DATASET_NAMES, y=[float(r["LR Test Acc"]) for r in results_rows]))
fig.update_layout(
    title="Probe Test Accuracy by Dataset",
    yaxis_title="Test Accuracy",
    yaxis_range=[0.5, 1.05],
    barmode="group",
    height=400,
    width=600,
)
fig.show()

# %%
DATASET_NAMES = ["cities", "sp_en_trans", "larger_than", "likely"]

# Load only the first 500 statements
datasets["likely"] = pd.read_csv(
    GOT_DATASETS / "likely.csv"
).head(500).copy()

# Clear any previously cached likely data
for cache in [
    activations, train_acts, test_acts,
    train_labels, test_labels,
    labels_dict, statements_dict,
]:
    cache.pop("likely", None)

df = datasets["likely"]
statements_dict["likely"] = df["statement"].tolist()
labels_dict["likely"] = t.tensor(df["label"].values, dtype=t.float32)

# Extract layer-14 activations with low GPU memory usage
gc.collect()
t.cuda.empty_cache()
model.eval()

acts_list = []

for statement in tqdm(statements_dict["likely"], desc="Extracting likely"):
    tokens = tokenizer(
        [statement],
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    last_token_idx = tokens["attention_mask"].sum(dim=1) - 1
    captured = []

    def hook_fn(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        idx = last_token_idx.to(hidden.device)
        acts = hidden[
            t.arange(hidden.shape[0], device=hidden.device),
            idx,
        ]
        captured.append(acts.detach().float().cpu())

    handle = model.model.layers[PROBE_LAYER].register_forward_hook(hook_fn)

    try:
        with t.inference_mode():
            model(
                **tokens,
                use_cache=False,
                output_hidden_states=False,
                logits_to_keep=1,
            )
    finally:
        handle.remove()

    acts_list.append(captured[0])
    del tokens, captured

activations["likely"] = t.cat(acts_list, dim=0)

# Same 80/20 train/test split as the other datasets
acts = activations["likely"]
labs = labels_dict["likely"]

n = len(acts)
perm = t.randperm(n, generator=t.Generator().manual_seed(42))
n_train = int(0.8 * n)

train_acts["likely"] = acts[perm[:n_train]]
test_acts["likely"] = acts[perm[n_train:]]
train_labels["likely"] = labs[perm[:n_train]]
test_labels["likely"] = labs[perm[n_train:]]

print(f"likely: train={n_train}, test={n - n_train}")
print(f"Activation shape: {acts.shape}")

# # %%
# # Compute 4x4 cross-dataset generalisation matrices

# DATASET_NAMES = ["cities", "sp_en_trans", "larger_than", "likely"]

# def compute_generalization_matrix(
#     train_acts, train_labels,
#     test_acts, test_labels,
#     dataset_names, probe_cls,
# ):
#     n = len(dataset_names)
#     matrix = t.zeros(n, n)

#     for i, train_name in enumerate(dataset_names):
#         probe = probe_cls.from_data(
#             train_acts[train_name],
#             train_labels[train_name],
#         )

#         for j, test_name in enumerate(dataset_names):
#             preds = probe.pred(test_acts[test_name])
#             matrix[i, j] = (
#                 (preds == test_labels[test_name])
#                 .float().mean().item()
#             )

#     return matrix


# mm_matrix = compute_generalization_matrix(
#     train_acts, train_labels, test_acts, test_labels,
#     DATASET_NAMES, MMProbe,
# )

# lr_matrix = compute_generalization_matrix(
#     train_acts, train_labels, test_acts, test_labels,
#     DATASET_NAMES, LRProbe,
# )

# assert mm_matrix.shape == (4, 4)
# assert lr_matrix.shape == (4, 4)


# # Plot both 4x4 matrices
# fig = make_subplots(
#     rows=1,
#     cols=2,
#     subplot_titles=["MMProbe", "LRProbe"],
#     horizontal_spacing=0.15,
# )

# for idx, matrix in enumerate([mm_matrix, lr_matrix]):
#     text_vals = [
#         [f"{matrix[i, j]:.3f}" for j in range(4)]
#         for i in range(4)
#     ]

#     fig.add_trace(
#         go.Heatmap(
#             z=matrix.numpy(),
#             x=DATASET_NAMES,
#             y=DATASET_NAMES,
#             text=text_vals,
#             texttemplate="%{text}",
#             colorscale="RdYlGn",
#             zmin=0.0,
#             zmax=1.0,
#             showscale=(idx == 1),
#         ),
#         row=1,
#         col=idx + 1,
#     )

#     fig.update_yaxes(
#         title_text="Train dataset" if idx == 0 else "",
#         row=1,
#         col=idx + 1,
#     )
#     fig.update_xaxes(
#         title_text="Test dataset",
#         tickangle=30,
#         row=1,
#         col=idx + 1,
#     )

# fig.update_layout(
#     title="Cross-dataset Generalization (Test Accuracy)",
#     height=500,
#     width=1100,
# )
# fig.show()


# # Cosine similarity between probe directions
# mm_directions = {
#     name: MMProbe.from_data(
#         train_acts[name], train_labels[name]
#     ).direction
#     for name in DATASET_NAMES
# }

# lr_directions = {
#     name: LRProbe.from_data(
#         train_acts[name], train_labels[name]
#     ).direction
#     for name in DATASET_NAMES
# }

# print("\nPairwise cosine similarity between probe directions:")

# for probe_name, directions in [
#     ("MM", mm_directions),
#     ("LR", lr_directions),
# ]:
#     print(f"\n  {probe_name}Probe:")

#     for i, n1 in enumerate(DATASET_NAMES):
#         for j, n2 in enumerate(DATASET_NAMES):
#             if j > i:
#                 d1 = directions[n1] / directions[n1].norm()
#                 d2 = directions[n2] / directions[n2].norm()

#                 print(
#                     f"    {n1} vs {n2}: "
#                     f"{(d1 @ d2).item():.4f}"
#                 )

# %%
# Few-shot prompt from geometry-of-truth/interventions.py. Note 3 true + 1 false examples:
# the paper calibrated this specific prompt for LLaMA-2-13B on sp_en_trans.
FEW_SHOT_PROMPT = """\
The Spanish word 'jirafa' means 'giraffe'. This statement is: TRUE
The Spanish word 'escribir' means 'to write'. This statement is: TRUE
The Spanish word 'gato' means 'cat'. This statement is: TRUE
The Spanish word 'aire' means 'silver'. This statement is: FALSE
"""

# Get token IDs for TRUE and FALSE
TRUE_ID = tokenizer.encode(" TRUE")[-1]
FALSE_ID = tokenizer.encode(" FALSE")[-1]

# %%
def few_shot_evaluate(
    statements: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    few_shot_prompt: str,
    true_id: int,
    false_id: int,
    batch_size: int = 32,
) -> Float[Tensor, " n"]:
    """
    Evaluate P(TRUE) - P(FALSE) for each statement using few-shot classification.

    Args:
        statements: List of statements to classify.
        model: Language model.
        tokenizer: Tokenizer.
        few_shot_prompt: The few-shot prefix prompt.
        true_id: Token ID for " TRUE".
        false_id: Token ID for " FALSE".
        batch_size: Batch size.

    Returns:
        Tensor of P(TRUE) - P(FALSE) for each statement.
    """
    p_diffs = []

    for i in range(0, len(statements), batch_size):
        batch = statements[i : i + batch_size]
        queries = [few_shot_prompt + stmt + " This statement is:" for stmt in batch]

        inputs = tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)

        with t.no_grad():
            outputs = model(**inputs)
            # Get logits at the last non-padding position
            last_idx = inputs["attention_mask"].sum(dim=1) - 1
            batch_indices = t.arange(len(batch), device=outputs.logits.device)
            last_logits = outputs.logits[batch_indices, last_idx]  # [batch, vocab]
            probs = last_logits.softmax(dim=-1)
            p_diff = probs[:, true_id] - probs[:, false_id]
            p_diffs.append(p_diff.cpu().float())

    return t.cat(p_diffs)

# Load sp_en_trans for evaluation (exclude statements used in the few-shot prompt)
sp_df = datasets["sp_en_trans"]
sp_statements = sp_df["statement"].tolist()
sp_labels = t.tensor(sp_df["label"].values, dtype=t.float32)

# Filter out statements that appear in the few-shot prompt
sp_eval_mask = [s not in FEW_SHOT_PROMPT for s in sp_statements]
sp_eval_stmts = [s for s, m in zip(sp_statements, sp_eval_mask) if m]
sp_eval_labels = sp_labels[t.tensor(sp_eval_mask)]

p_diffs = few_shot_evaluate(sp_eval_stmts, model, tokenizer, FEW_SHOT_PROMPT, TRUE_ID, FALSE_ID)

# Compute accuracy
preds = (p_diffs > 0).float()
acc = (preds == sp_eval_labels).float().mean().item()
assert acc > 0.9, f"Few-shot accuracy too low: {acc:.3f} (expected > 0.9)"
true_mean = p_diffs[sp_eval_labels == 1].mean().item()
false_mean = p_diffs[sp_eval_labels == 0].mean().item()

print(f"Few-shot classification accuracy: {acc:.3f}")
print(f"Mean P(TRUE)-P(FALSE) for true statements:  {true_mean:.4f}")
print(f"Mean P(TRUE)-P(FALSE) for false statements: {false_mean:.4f}")

# Histogram
fig = go.Figure()
fig.add_trace(
    go.Histogram(x=p_diffs[sp_eval_labels == 1].numpy(), name="True", marker_color="blue", opacity=0.6, nbinsx=30)
)
fig.add_trace(
    go.Histogram(x=p_diffs[sp_eval_labels == 0].numpy(), name="False", marker_color="red", opacity=0.6, nbinsx=30)
)
fig.add_vline(x=0, line_dash="dash", line_color="gray")
fig.update_layout(
    title="Few-Shot Classification: P(TRUE) - P(FALSE)",
    xaxis_title="P(TRUE) - P(FALSE)",
    yaxis_title="Count",
    barmode="overlay",
    height=400,
    width=700,
)
fig.show()

# %%
def make_intervention_hook(
    direction: Float[Tensor, " d_model"],
    scale: float,
    positions: list[int],
) -> callable:
    """
    Create a forward hook that adds scale * direction to hidden states at fixed positions.
    This handles both plain-tensor and tuple outputs from transformer layers.
    """

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        for pos in positions:
            if 0 <= pos < hidden_states.shape[1]:
                hidden_states[:, pos, :] += scale * direction

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        else:
            return hidden_states

    return hook_fn

# %%
def intervention_experiment(
    statements: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    direction: Float[Tensor, " d_model"],
    few_shot_prompt: str,
    true_id: int,
    false_id: int,
    intervene_layers: list[int],
    intervention: str = "none",
    batch_size: int = 32,
) -> Float[Tensor, " n"]:
    """
    Run the intervention experiment.

    Args:
        statements: Statements to evaluate.
        model: Language model.
        tokenizer: Tokenizer.
        direction: The (already scaled) truth direction vector.
        few_shot_prompt: Few-shot prefix.
        true_id: Token ID for " TRUE".
        false_id: Token ID for " FALSE".
        intervene_layers: List of layer indices to intervene at.
        intervention: "none", "add", or "subtract".
        batch_size: Batch size.

    Returns:
        P(TRUE) - P(FALSE) for each statement.
    """
    assert intervention in ["none", "add", "subtract"]

    # Determine how many tokens " This statement is:" adds
    suffix_tokens = tokenizer.encode(" This statement is:")
    len_suffix = len(suffix_tokens)

    p_diffs = []
    for i in range(0, len(statements), batch_size):
        batch = statements[i : i + batch_size]
        queries = [few_shot_prompt + stmt + " This statement is:" for stmt in batch]

        inputs = tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)

        # Register hooks for intervention
        hooks = []
        if intervention != "none":
            dir_device = direction.to(model.device)
            scale = 1.0 if intervention == "add" else -1.0

            # Each sequence in the batch can have a different length, so we iterate over batch
            # elements inside the hook, using attention_mask to find real sequence lengths.
            def make_batch_hook(dir_vec, attn_mask, scl):
                def hook_fn(module, input, output):
                    hidden_states = output[0] if isinstance(output, tuple) else output

                    seq_lens = attn_mask.sum(dim=1)  # [batch]
                    for b in range(hidden_states.shape[0]):
                        end = seq_lens[b].item()
                        for offset in [-len_suffix, -len_suffix - 1]:
                            pos = int(end + offset)
                            if 0 <= pos < hidden_states.shape[1]:
                                hidden_states[b, pos, :] += scl * dir_vec

                    return (hidden_states,) + output[1:] if isinstance(output, tuple) else hidden_states

                return hook_fn

            for layer_idx in intervene_layers:
                hook = model.model.layers[layer_idx].register_forward_hook(
                    make_batch_hook(dir_device, inputs["attention_mask"], scale)
                )
                hooks.append(hook)

        with t.no_grad():
            # Common pattern for hooks, so failed hooks don't get stuck
            try:
                outputs = model(**inputs)
            finally:
                for hook in hooks:
                    hook.remove()

            # Get logits at the last non-padding position, then get probability differences
            last_idx = inputs["attention_mask"].sum(dim=1) - 1
            batch_indices = t.arange(len(batch), device=outputs.logits.device)
            last_logits = outputs.logits[batch_indices, last_idx]
            probs = last_logits.softmax(dim=-1)
            p_diff = probs[:, true_id] - probs[:, false_id]
            p_diffs.append(p_diff.cpu().float())

    return t.cat(p_diffs)


# Train the intervention probe on cities + neg_cities combined. The paper found that
# "training on statements and their opposites improves generalization" - using both
# a statement and its negation gives the probe a cleaner truth direction.
# Load neg_cities for this paired training
neg_cities_df = pd.read_csv(GOT_DATASETS / "neg_cities.csv")
neg_cities_stmts = neg_cities_df["statement"].tolist()
neg_cities_labels = t.tensor(neg_cities_df["label"].values, dtype=t.float32)

neg_cities_acts_dict = extract_activations(neg_cities_stmts, model, tokenizer, [PROBE_LAYER])
neg_cities_acts = neg_cities_acts_dict[PROBE_LAYER]

# Train probe on cities + neg_cities combined
combined_acts = t.cat([activations["cities"], neg_cities_acts])
combined_labels = t.cat([labels_dict["cities"], neg_cities_labels])
combined_probe = MMProbe.from_data(combined_acts, combined_labels)

# Scale the direction
direction = combined_probe.direction
direction_hat = direction / direction.norm()
true_acts = combined_acts[combined_labels == 1]
false_acts = combined_acts[combined_labels == 0]
true_mean = true_acts.mean(0)
false_mean = false_acts.mean(0)
projection_diff = ((true_mean - false_mean) @ direction_hat).item()
scaled_direction = projection_diff * direction_hat

# Intervene at all layers from INTERVENE_LAYER through PROBE_LAYER. This matches
# the paper's "group (b)" hidden states that were found to be causally implicated.
intervene_layer_list = list(range(INTERVENE_LAYER, PROBE_LAYER + 1))

# Run for all 3 conditions × 2 subsets
results_intervention = {}
for intervention_type in ["none", "add", "subtract"]:
    for subset in ["true", "false"]:
        mask = sp_eval_labels == (1 if subset == "true" else 0)
        subset_stmts = [s for s, m in zip(sp_eval_stmts, mask.tolist()) if m]
        p_diffs = intervention_experiment(
            subset_stmts,
            model,
            tokenizer,
            scaled_direction,
            FEW_SHOT_PROMPT,
            TRUE_ID,
            FALSE_ID,
            intervene_layer_list,
            intervention=intervention_type,
        )
        results_intervention[(intervention_type, subset)] = p_diffs.mean().item()

# Print results
intervention_df = pd.DataFrame(
    {
        "Intervention": ["none", "add", "subtract"],
        "True Stmts (mean P_diff)": [
            f"{results_intervention[('none', 'true')]:.4f}",
            f"{results_intervention[('add', 'true')]:.4f}",
            f"{results_intervention[('subtract', 'true')]:.4f}",
        ],
        "False Stmts (mean P_diff)": [
            f"{results_intervention[('none', 'false')]:.4f}",
            f"{results_intervention[('add', 'false')]:.4f}",
            f"{results_intervention[('subtract', 'false')]:.4f}",
        ],
    }
)
print("\nIntervention results (mean P(TRUE) - P(FALSE)):")
display(intervention_df)

# Grouped bar chart
fig = go.Figure()
for subset, color in [("true", "blue"), ("false", "red")]:
    vals = [results_intervention[(interv, subset)] for interv in ["none", "add", "subtract"]]
    fig.add_trace(
        go.Bar(
            name=f"{subset.capitalize()} statements",
            x=["None", "Add", "Subtract"],
            y=vals,
            marker_color=color,
            opacity=0.7,
        )
    )
fig.update_layout(
    title="Causal Intervention: Effect on P(TRUE) - P(FALSE)",
    yaxis_title="Mean P(TRUE) - P(FALSE)",
    barmode="group",
    height=400,
    width=600,
)
fig.add_hline(y=0, line_dash="dash", line_color="gray")
fig.show()

# %%
