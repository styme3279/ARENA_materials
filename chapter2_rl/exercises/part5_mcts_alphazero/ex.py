#%%

import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

chapter = "chapter2_rl"
repo = "ARENA_materials"
branch = "main"

# Install dependencies (Colab already has torch, einops, numpy, matplotlib, plotly, tqdm and huggingface_hub)
try:
    import eindex, jaxtyping, torchinfo
except:
    %pip install einops jaxtyping torchinfo git+https://github.com/ARENA-education/eindex.git

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

try:
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
except Exception:
    pass
import einops
from eindex import eindex
import math
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from jaxtyping import Float, Bool, Int
from torch import Tensor
import torch as t  
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from tqdm.auto import tqdm

# Make sure exercises are in the path
chapter = "chapter2_rl"
section = "part5_mcts_alphazero"
exercises_dir = next(p for p in Path.cwd().parents if (p / chapter).exists()) / chapter / "exercises"
section_dir = exercises_dir / section
for _p in (str(section_dir), str(exercises_dir)):   # section_dir for bare imports; exercises_dir so
    if _p not in sys.path:                          # `from part5_mcts_alphazero.solutions import ...` resolves
        sys.path.append(_p)
import part5_mcts_alphazero.tests as tests
import part5_mcts_alphazero.utils as utils
from part5_mcts_alphazero.utils import (
    Board, Connect4Env, MCTSConfig, legal_mask_from_obs, fmt_si,
    render_board, place_piece, plot_board_and_policy, print_mcts_tree, plot_mcts_tree,
    two_ply_positions, greedy_policy_action,
)

# The single-game MCTS runs lots of tiny CPU tensor ops in a Python loop. With PyTorch's default of one
# thread per core, its threads spend longer coordinating than computing, and on cloud machines with a
# CPU quota they can get the whole process throttled (we measured a ~40x slowdown). Adjust if needed.
t.set_num_threads(min(4, t.get_num_threads()))
t.set_float32_matmul_precision('high')   # allow TF32 matmuls on Ampere+ GPUs: faster training, ~1e-3 relative precision

device = t.device("cuda" if t.cuda.is_available() else "mps" if t.backends.mps.is_available() else "cpu")
MAIN = __name__ == "__main__"
SLOW = False   # set True to run the slow bonus demos (strength-vs-sims, Elo-vs-search budget)
TRAINING = True    # set False to skip section 4 self-play training (e.g. while iterating on section 2/section 3)
# %%
