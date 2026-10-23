# %%

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
env = Connect4Env(device=device)
board = env.reset(1)
board, _, _ = env.step(board, t.tensor([3], device=device))   # red plays column 3
board, _, _ = env.step(board, t.tensor([3], device=device))   # yellow replies in column 3
print(render_board(board))
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
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels,kernel_size=3,padding=1,bias=False),        
            nn.BatchNorm2d(num_features=channels),
        )
        self.ReLU = nn.ReLU()
        

    def forward(self, x: Float[Tensor, "B C H W"]) -> Float[Tensor, "B C H W"]:
        """Two conv-BN layers (ReLU between), then add the input back (skip) and ReLU.

        Args:
            x: (B, C, H, W) input feature map

        Returns:
            (B, C, H, W) output feature map (shape preserved)
        """
        return self.ReLU(x + self.layers(x))


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
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=conv_out,kernel_size=1,bias=False),
            nn.BatchNorm2d(num_features=conv_out),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv_out*height*width,32),
            nn.ReLU(),
            nn.Linear(32,1),
            nn.Tanh()
        )

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
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=conv_out,kernel_size=1,bias=False),
            nn.BatchNorm2d(num_features=conv_out),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv_out*height*width,7)
        )
        
    def forward(self, x: Float[Tensor, "B C H W"]) -> Float[Tensor, "B 7"]:
        """Map the shared trunk to one policy logit per column.

        Args:
            x: (B, C, 6, 7) shared-trunk features

        Returns:
            (B, 7) one logit per column
        """
        return self.net(x)


tests.test_actor(Actor)
# %%
class Connect4Model(nn.Module):
    """The full AlphaZero network: a shared convolutional trunk and the two heads.

    Architecture (build the modules in this order — see note below):
    - `self.features` (the trunk): Conv2d 3×3 `3 -> channels` (padding=1, bias=False) -> BatchNorm
      -> ReLU, then two `ResBlock(channels)`.
    - `self.critic = Critic(channels, critic_conv_out, height, width)` on the trunk output.
    - `self.actor = Actor(channels, actor_conv_out, height, width)` on the trunk output.

    Note: create trunk, then critic, then actor, in that order. The tests check your model is
    functionally identical to the reference by copying the reference weights in by
    *parameter-creation order*, so a correct architecture built in a different order fails them.
    """
    def __init__(self,
                 device,
                 channels: int = 128,
                 critic_conv_out: int = 3,
                 actor_conv_out: int = 32,
                 height: int = 6,
                 width: int = 7,
    ):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU()
        )
        self.block2 = ResBlock(channels)
        self.block3 = ResBlock(channels)
        self.critic = Critic(in_channels=channels,conv_out=critic_conv_out,height=height,width=width)
        self.actor = Actor(in_channels=channels,conv_out=actor_conv_out,height=height,width=width)
        self.to(device)

    def forward(
        self, boards: Board
    ) -> tuple[Float[Tensor, "B"], Float[Tensor, "B 7"]]:
        """Run the shared trunk then both heads on a batch of boards.

        Args:
            boards: a `Board` of batch shape (B,).

        Returns:
            value:  (B,) the position's value for the mover, in [-1, 1] (tanh-squashed)
            logits: (B, 7) one policy logit per column
        """
        x = boards.mover_view()   # (B, C, H, W)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        value =self.critic(x)
        logits = self.actor(x)
        return value, logits


summary(Connect4Model(device))
tests.test_connect4_model(Connect4Model)
# %%
