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
env = Connect4Env(device=device)
board = env.reset(1)
board, _, _ = env.step(board, t.tensor([3], device=device))   # red plays column 3
board, _, _ = env.step(board, t.tensor([3], device=device))   # yellow replies in column 3
print(render_board(board))
# %%
