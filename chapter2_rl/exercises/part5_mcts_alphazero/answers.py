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

boards = env.reset(4)                    # a Board of shape (4,): 4 empty boards, red to move
boards.obs.shape                         # t.Size([4, 3, 6, 7])
boards.red_to_move                       # tensor([True, True, True, True])
boards[1:3]                              # a Board of shape (2,)

boards, done, reward = env.step(boards, t.tensor([3, 3, 0, 6]))   # one move in every game
boards.red_to_move                       # tensor([False, False, False, False]): yellow to move
boards[[0, 2]] = env.reset(2)            # restart games 0 and 2: writes their obs AND red_to_move
boards.red_to_move                       # tensor([ True, False,  True, False])

# %%

board = env.reset(1)
board, _, _ = env.step(board, 3)   # red plays column 3 (env.step flips red_to_move for us)
print(render_board(board), "\n")
print("...and from the mover's (yellow's) perspective, as the network sees it:")
print(render_board(Board(board.mover_view(), board.red_to_move), show_to_move=False))
# channels swapped: red's piece is now the opponent's, so it is drawn as O


# %%

class ResBlock(nn.Module):
    """A residual block, shape-preserving on (B, channels, H, W):

        x -> Conv2d 3×3 -> BatchNorm -> ReLU -> Conv2d 3×3 -> BatchNorm -> (+ x) -> ReLU

    Both convs are `channels -> channels`, 3×3, padding=1, `bias=False`: each conv is followed
    immediately by a BatchNorm, whose learned shift makes a conv bias redundant.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )
        

    def forward(self, x: Float[Tensor, "B C H W"]) -> Float[Tensor, "B C H W"]:
        """Two conv-BN layers (ReLU between), then add the input back (skip) and ReLU.

        Args:
            x: (B, C, H, W) input feature map

        Returns:
            (B, C, H, W) output feature map (shape preserved)
        """
        return nn.ReLU(x + self.res(x))


tests.test_resblock(ResBlock)

# %%
class Critic(nn.Module):
    """The value head, (B, in_channels, H, W) trunk features -> (B,) values in [-1, 1]:

        Conv2d 1×1 `in_channels -> conv_out` (bias=False) -> BatchNorm -> ReLU -> Flatten
        -> Linear(conv_out*height*width -> 32) -> ReLU -> Linear(32 -> 1) -> Tanh -> squeeze to (B,)

    The 1×1 conv shrinks the trunk to `conv_out` channels before flattening (a per-square shared
    Linear), and the final tanh bounds the value to [-1, 1] to match the game-outcome targets z.
    As in `ResBlock`, the conv uses `bias=False`: the BatchNorm right after makes a conv bias redundant.
    """
    def __init__(self, in_channels=128, conv_out=3, height=6, width=7):
        super().__init__()
        self.critic = 

    def forward(self, x: Float[Tensor, "B C H W"]) -> Float[Tensor, "B"]:
        """Map the shared trunk to a scalar value for the side to move.

        Args:
            x: (B, C, 6, 7) shared-trunk features

        Returns:
            (B,) the position's value for the mover, in [-1, 1] (tanh-squashed)
        """
        return self.net(x).squeeze(-1)  # (B, 1) -> (B,)


tests.test_critic(Critic)


# %%

class Actor(nn.Module):
    """The policy head, (B, in_channels, H, W) trunk features -> (B, width) column logits:

        Conv2d 1×1 `in_channels -> conv_out` (bias=False) -> BatchNorm -> ReLU -> Flatten
        -> Linear(conv_out*height*width -> width)

    Same shrink-then-flatten pattern as the Critic (note the different `conv_out` default: 32
    here vs 3 there), but the final Linear emits one logit per column and there is no squashing —
    these are raw logits, softmaxed later. Conv bias=False (a BatchNorm follows), as in the Critic.
    """
    def __init__(self, in_channels=128, conv_out=32, height=6, width=7):
        super().__init__()
        self.actor = 

    def forward(self, x: Float[Tensor, "B C H W"]) -> Float[Tensor, "B 7"]:
        """Map the shared trunk to one policy logit per column.

        Args:
            x: (B, C, 6, 7) shared-trunk features

        Returns:
            (B, 7) one logit per column
        """
        return self.net(x)


tests.test_actor(Actor)
