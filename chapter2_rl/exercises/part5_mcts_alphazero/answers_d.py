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

Action = int # 0, 1, 2, 3, 4, 5, 6

@dataclass
class Node:
    """One node of the single-game search tree = one board state. Statistics live on the **edges**
    (per-action length-7 arrays), filled in as the search runs.

    Attributes:
        board          Board of batch shape (1,): the connect4 board for this state, and whose turn
                       it is. Note that `Connect4Env.step` auto-resets a finished board, so on a **terminal**
                       node this is the empty starting board, for which
                       only `terminal_value` is well-defined.
        is_terminal    bool : True if this state is game-over. Set by `expand`.
        terminal_value float : result from the perspective of this node's mover (= -reward of the
                       move that created the node). Only meaningful when `is_terminal=True`. Set by `expand`.
        P              (7,) network prior over each move. Set by `evaluate`.
        legal          (7,) bool legal-column mask. Set by `evaluate`.
        N              (7,) per-edge visit counts N(s, a). Updated by `backup`.
        W              (7,) per-edge value sums W(s, a), accumulated from the perspective of THIS
                       node's mover: backup negates each child value once per step on
                       the way up, so what lands here is already in this node's convention. Updated by `backup`.
        children       dict[Action, Node] : child nodes per played action, created lazily by `expand`.
        parent         Node | None : the node we were expanded from (`None` at the root).
        parent_action  Action | None : the column that led from `parent` to this node. Updated by `expand`.

    Properties:
        Q              (7,) per-edge mean value W / max(N, 1).
    """
    board: Board  # batch shape (1,); lives on GPU
    is_terminal: bool = False
    terminal_value: float = 0.0
    P: Tensor | None = None
    legal: Tensor | None = None
    N: Float[Tensor, "7"] = field(default_factory=lambda: t.zeros(7))   # N(s,a_0), ..., N(s,a_6)
    W: Float[Tensor, "7"] = field(default_factory=lambda: t.zeros(7))   # W(s,a_0), ..., W(s,a_6)
    children: dict[Action, 'Node'] = field(default_factory=dict)
    parent: 'Node | None' = None
    parent_action: Action | None = None

    @property
    def Q(self) -> Float[Tensor, "7"]:
        return self.W / t.maximum(self.N, t.ones_like(self.N))


# Sentinel "no action" returned by `select` when the selection walk stops on a terminal node.
# We use -100 rather than None because None silently inserts a dim when used to index a tensor
# (`t[None]`), whereas -100 is always an illegal column and raises loudly if ever misused as an index.
NULL_ACTION = -100

tests.test_mcts_node(Node)
# %%

def select_child(node : Node, c_puct: float) -> Action:
    """Return the action (column index) with the highest PUCT score at `node`.

    Everything you need is already stored on the node (see the `Node` class above):
        node.Q      (7,) per-edge mean values Q(s,a) = W / max(N, 1)  — a property, kept up to date
        node.N      (7,) per-edge visit counts N(s,a); its sum is the parent visit total Σ_a' N(s,a')
        node.P      (7,) the network prior p_θ(a|s), set when the node was evaluated
        node.legal  (7,) bool mask of legal columns

    Compute the PUCT score per action, set illegal actions to -inf, and take the argmax.

    Args:
        node:   the node to pick an action from (its edge statistics are already populated)
        c_puct: the exploration/exploitation trade-off constant c

    Returns:
        int: the PUCT-maximising column 0-6 (you may assume at least one action is legal).
        Return a plain Python int — `int(scores.argmax())`, not a 0-dim Tensor. A Tensor looks
        identical in comparisons but breaks `action in node.children` dict lookups in `select`
        (tensors hash by identity, not value).

        PUCT(s,a) = Q(s,a) + c_puct * p(a|s) * sqrt(1 + sum(N(s,a'))) / (1 + N(s,a))
    """
    Q = node.Q
    p = node.P
    num_N = t.sqrt(1 + node.N.sum())
    denom_N = node.N + 1

    puct = Q + c_puct * p * num_N / denom_N 
    legal_score = t.where(node.legal,puct,-t.inf)

    return int(legal_score.argmax())


tests.test_select_child(select_child, Node)
# %%
def select(root: Node, 
           c_puct: float
) -> tuple[Node, Action]:
    """Walks down the tree from root: at each node take the PUCT-best action (`select_child`);
    if that action has a child, step into it and repeat. Halt at the first node whose PUCT-best
    action has NO child yet (`action not in node.children`), or on a terminal node.

    Note the halting rule is a property of the node's best *action*, not of the node itself: a
    node may already have children on other actions (and visits on them) while its current
    PUCT-best action is still unexpanded — that node is where the walk stops.

    Args:
        root: The root node of the search tree (never terminal).
        c_puct: The PUCT constant.

    Returns:
        * `(node, action)`: the node to expand and the action to play, OR
        * `(node, NULL_ACTION)` if the walk stopped on a terminal node (nothing to expand).

    `env.reset` returns `Board`s, and `env.step(boards, actions)` returns the next `Board`s (with
    `red_to_move` flipped, since the opponent moves next), along with `done` and `reward`.
    """
    assert not root.is_terminal, "select: root must not be terminal"

    curr_node = root

    while not curr_node.is_terminal:
        a = select_child(curr_node,c_puct)
        if a not in curr_node.children:
            return curr_node, a
        curr_node = curr_node.children[a]
    return curr_node, NULL_ACTION

tests.test_select(select, Node)
# %%

@t.no_grad()
def expand(node: Node, 
           action: int, 
           env: Connect4Env) -> Node:
    """
    Plays the given action on the current board state, constructs a new node with the result,
    stores it as a child of the current node, and returns it.

    Side effects:
        * Mutates the current node to add a new child node.

    Args:
        node: The current node in the search tree.
        action: The action to play.
        env: The Connect4Env instance

    Returns:
        The newly created child node after the action has been played.

        board          Board of batch shape (1,): the connect4 board for this state, and whose turn
                       it is. Note that `Connect4Env.step` auto-resets a finished board, so on a **terminal**
                       node this is the empty starting board, for which
                       only `terminal_value` is well-defined.
        is_terminal    bool : True if this state is game-over. Set by `expand`.
        terminal_value float : result from the perspective of this node's mover (= -reward of the
                       move that created the node). Only meaningful when `is_terminal=True`. Set by `expand`.
        P              (7,) network prior over each move. Set by `evaluate`.
        legal          (7,) bool legal-column mask. Set by `evaluate`.
        N              (7,) per-edge visit counts N(s, a). Updated by `backup`.
        W              (7,) per-edge value sums W(s, a), accumulated from the perspective of THIS
                       node's mover: backup negates each child value once per step on
                       the way up, so what lands here is already in this node's convention. Updated by `backup`.
        children       dict[Action, Node] : child nodes per played action, created lazily by `expand`.
        parent         Node | None : the node we were expanded from (`None` at the root).
        parent_action  Action | None : the column that led from `parent` to this node. Updated by `expand`.
    """
    assert not node.is_terminal, "expand: cannot expand a terminal node"

    next_board, done, reward = env.step(node.board,action)

    child = Node(
        board=next_board,
        is_terminal=done,
        terminal_value=-reward,
        parent=node,
        parent_action=action
    )
    node.children[action] = child

    return child


tests.test_expand(expand)
# %%

@t.no_grad()
def evaluate(node: Node, model: nn.Module, env: Connect4Env) -> float:
    """
    Estimates the value of a node from its mover's perspective.
    
    Side effects:
        If the node is non-terminal, mutates the node to 
        set the `node.P` and `node.legal` attributes to the
        action probability from the model, and to the current legal moves
        of the board respectively.
    
    Args:
        node: The node to estimate the value of.
        model: The model to use.
        env: The Connect4Env instance

    Returns:
        For terminal nodes: the stored `terminal_value` (already from the mover's perspective).
        For non-terminal nodes: the critic head's value estimate, from the mover's perspective.
    """
    if node.is_terminal:
        return node.terminal_value

    value, logits = model(node.board) 
    value = value.squeeze(0).float().cpu()
    logits = logits.squeeze(0).float().cpu()

    node.legal = env.legal_action_mask(boards=node.board).squeeze(0).cpu()
    legal_logits = t.where(node.legal,logits,-t.inf)
    node.P = t.softmax(legal_logits, dim=-1)


    return float(value)


tests.test_evaluate(evaluate)
# %%
def backup(leaf: Node, value: float) -> None:
    """Walks from a node back up the tree to the root, updating the visit counts and value sums at each edge.
    
    Side effects:
        * Mutates the visit counts and value sums at each edge.
    
    Args:
        leaf: The leaf node to backup from.
        value: The value of the leaf node from its mover's perspective.

    Returns:
        None
    """
    v, curr = value, leaf
    while curr.parent is not None:
        v = -v
        curr.parent.N[curr.parent_action] += 1.0
        curr.parent.W[curr.parent_action] += v
        curr = curr.parent


tests.test_backup(backup, Node)
# %%
def sample_tree_policy(visits: Float[Tensor, "7"], 
                       temperature: float = 1.0,
) -> Action:
    """Samples an action from the tree policy. `temperature` -> 0 is greedy argmax; `temperature` = 1
    samples proportional to visits; larger temperatures flatten the distribution.
    
    Args:
        visits: The visit counts for each action, shape `(7,)`.
        temperature: The temperature parameter.

    Returns:
        int: The action to sample.
    """
    if temperature < 1e-8:
        return int(visits.argmax())
    
    numerator = visits**(1/temperature) 
    probs = numerator / numerator.sum()

    return int(t.multinomial(probs,1))
    


tests.test_sample_tree_policy(sample_tree_policy)
# %%
