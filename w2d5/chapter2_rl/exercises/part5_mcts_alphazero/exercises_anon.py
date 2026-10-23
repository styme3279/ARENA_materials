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
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x: Float[Tensor, "B C H W"]) -> Float[Tensor, "B C H W"]:
        """Two conv-BN layers (ReLU between), then add the input back (skip) and ReLU.

        Args:
            x: (B, C, H, W) input feature map

        Returns:
            (B, C, H, W) output feature map (shape preserved)
        """
        return nn.ReLU()(self.layers(x) + x)


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
            nn.Conv2d(in_channels, conv_out, 1, bias=False),
            nn.BatchNorm2d(conv_out),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv_out*height*width, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
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
            nn.Conv2d(in_channels, conv_out, 1, bias=False),
            nn.BatchNorm2d(conv_out),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv_out*height*width, 7),
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
        self.trunk = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            ResBlock(channels),
            ResBlock(channels),
        )
        self.critic = Critic(channels, critic_conv_out, height, width)
        self.actor = Actor(channels, actor_conv_out, height, width)
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
        x = self.trunk(x)
        critic = self.critic(x)
        actor = self.actor(x)
        return critic, actor


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
    """
    assert node is not None, "select_child: node must not be None"
    scores = (
            node.Q + 
            c_puct * node.P *
            (
                (1 + node.N.sum()).sqrt() / 
                (1 + node.N)
            )
        )
    return int(t.where(node.legal, scores, -float('inf')).argmax())


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
    """
    assert not root.is_terminal, "select: root must not be terminal"
    node = root
    while True:
        # print("visiting node ", node)
        if node.is_terminal:
            return (node, NULL_ACTION)
        action = select_child(node, c_puct)
        if action not in node.children:
            return (node, action)
        node = node.children[action]
    
        


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
    """
    assert not node.is_terminal, "expand: cannot expand a terminal node"
    board, done, reward = env.step(node.board, action)
    new = Node(board, done, -reward)
    new.parent = node
    new.parent_action = action
    node.children[action] = new
    return new


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
    probs = t.softmax(logits, dim=1).squeeze(0)
    node.P = probs.cpu()
    node.legal = env.legal_action_mask(node.board).squeeze(0).cpu()
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
    v, node = value, leaf
    while node.parent is not None:
        v = -v
        node.parent.W[node.parent_action] += v
        node.parent.N[node.parent_action] += 1
        node = node.parent
    return


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
    if temperature <= 1e-8:
        return int(visits.argmax())
    distr = (visits.pow(1/temperature) / visits.pow(1/temperature).sum())
    return int(t.multinomial(distr, 1))


tests.test_sample_tree_policy(sample_tree_policy)
# %%
@t.no_grad()
def mcts_search(
    root_board: Board,
    model: nn.Module,
    env: Connect4Env,
    cfg: MCTSConfig,
    add_noise: bool = False,
) -> Node:
    """Runs single-game MCTS from `root_board` and returns the root of the search tree.

    The root node is created and evaluated (setting its prior `P` and `legal` mask) before the
    loop. Each of the `cfg.sims` simulations then grows the tree by one node:
    SELECT a leaf (`select`), EXPAND it if it is not terminal (`expand`), EVALUATE the leaf
    (`evaluate`), and BACKUP its value to the root (`backup`).

    Args:
        root_board:  the position to search from, a single-game `Board` of batch shape (1,).
                     Must not be terminal.
        model:       the policy-value network, called as `model(boards) -> (value, logits)`.
        env:         the Connect4Env, used to play moves during expansion.
        cfg:         search settings; reads `cfg.sims` and `cfg.c_puct` (plus `cfg.dirichlet_alpha`
                     and `cfg.dirichlet_eps` when `add_noise=True`).
        add_noise:   if True, mix Dirichlet exploration noise into the root prior (self-play only;
                     leave False for evaluation, so the search is deterministic).

    Returns:
        The root `Node`. Its `root.N` are the (7,) root visit counts N(root, a), summing to
        `cfg.sims` (illegal columns get 0); its `children` hold the rest of the tree.
    """
    root = Node(root_board)
    evaluate(root, model, env) #required to set root.P and root.legal
    if add_noise:   # mix Dirichlet exploration noise into the root prior (`dirichlet_root_noise` is
                    # given a few cells below; every call before then passes add_noise=False)
        root.P = dirichlet_root_noise(root.P, root.legal, cfg.dirichlet_alpha, cfg.dirichlet_eps)

    for _ in range(cfg.sims):
        node = root
        leaf, action = select(node, cfg.c_puct)
        if not leaf.is_terminal:
            leaf = expand(leaf, action, env) # we go to the new leaf
        value = evaluate(leaf, model, env)
        backup(leaf, value)
    return root

# First check the search logic in isolation, with a dummy (uniform-policy, zero-value) network:
# a forced win-in-one must be found purely from the terminal reward backing up the tree.
tests.test_mcts_search(mcts_search)
# Then confirm the same search drives the real network correctly:
tests.test_mcts_search(mcts_search, Connect4Model(device).eval())
# %%
class DummyNet(nn.Module):

    def forward(self, boards):
        b, dev = boards.shape[0], boards.obs.device
        values = t.zeros(b, device=dev)
        logits = t.zeros(b, 7, device=dev)
        return values, logits

model = DummyNet()
board = tests.diagonal_win_red()

print("Starting position (X = Red to move):")
print(render_board(board))

root = mcts_search(board, model, env, MCTSConfig(sims=32))
visits = root.N
print("\nMCTS visit counts per column:", visits.int().tolist())
chosen = int(visits.argmax())
print(f"Most-visited column: {chosen}  ({int(visits[chosen])} of {int(visits.sum())} visits)")

board_after = place_piece(board, chosen)
print(f"\nBoard after X plays column {chosen}  (completes the diagonal):")
print(render_board(board_after, show_to_move=False))

# board + the visit-count policy pi(a) = N(s,a) / sum_a' N(s,a'), chosen column highlighted
plot_board_and_policy(board, visits / visits.sum(), chosen_action=chosen,
                      title="MCTS finds the diagonal win")
# the search tree the simulations grew: edge thickness ~ visit count, terminal leaves in yellow
plot_mcts_tree(root, max_depth=2, title="MCTS search tree (after 32 sims)")
assert chosen == 4, "MCTS should find the diagonal win"
# %%
def dirichlet_root_noise(
    prior: Float[Tensor, "... 7"],
    legal: Bool[Tensor, "... 7"],
    alpha: float,
    eps: float,
) -> Float[Tensor, "... 7"]:
    """Mix Dirichlet exploration noise into the root prior (used by `expand_root` when `add_noise`).

    Noise is added only at the root, which keeps self-play exploring without distorting the rest of
    the tree. `eps = 0` returns `prior` unchanged. We use a symmetric Dirichlet (the same `alpha`
    for every column). Works with or without a leading batch dimension.

    Args:
        prior: (..., 7) the network prior at the root
        legal: (..., 7) legal-column mask (the noise is renormalised over the legal columns)
        alpha: Dirichlet concentration (smaller = spikier noise)
        eps:   mixing weight on the noise

    Returns:
        (..., 7) the mixed prior `(1 - eps) * prior + eps * noise`
    """
    noise = t.distributions.Dirichlet(
        t.full((prior.shape[-1],), alpha, device=prior.device)
    ).sample(prior.shape[:-1])
    noise = noise * legal.float()
    noise = noise / noise.sum(-1, keepdim=True).clamp_min(1e-8)
    return (1.0 - eps) * prior + eps * noise


tests.test_dirichlet_root_noise(dirichlet_root_noise)
# %%
utils.plot_dirichlet_simplex()

# %%
class SimulatedBatchedMCTS:
    """Root-parallel MCTS over `B` independent trees, with the network call batched across trees.

    A clarity-first stand-in for the vectorised `BatchedMCTS` below, with the **same interface**: hold
    an `env` + `cfg`, then call `.search(model, root_boards)`. Every tree is a normal
    Python `Node` tree driven by the section 2 `select`/`backup` functions, looped over the batch. The two
    things it borrows from the vectorised version are (i) a single board `pool` that every node indexes
    by `slot` (so boards never need packing/unpacking), and (ii) a fixed-size forward pass over all
    `B` leaves every simulation, terminal leaves included.

    Each node stores an integer `slot` instead of its own board: its position lives at
    `pool[game, slot]`. (`Node.board` is left as `None` since none of `select`/`backup` read it.)
    """

    def __init__(self, env, cfg):
        self.env, self.cfg = env, cfg

    @t.no_grad()
    def _expand(self, pool, nptr, game, node, action):
        """Section 2's `expand` function, but the child's board is written into the pool and the child stores its `slot`."""
        parent_board = pool[game, node.slot]                                # one game's board, unbatched
        next_board, done, reward = self.env.step(parent_board, action)      # done, reward: python scalars
        slot = nptr[game]
        nptr[game] += 1
        pool[game, slot] = next_board
        child = Node(board=None, is_terminal=done, terminal_value=-reward,
                     parent=node, parent_action=action)
        child.slot = slot
        node.children[action] = child
        return child

    @t.no_grad()
    def _evaluate_batch(self, model, pool, batch_idx, nodes):
        """Evaluate all `B` leaves in ONE fixed-size forward pass; return one value per node.

        Terminal leaves are forwarded too (their board is gathered from the pool like any other) even
        though we throw away the network's output and use the value stored at creation. The constant
        `B`-shaped batch matches the vectorised search and is faster than a ragged one.
        """
        slots = t.tensor([node.slot for node in nodes], device=batch_idx.device)
        boards = pool[batch_idx, slots]                                    # (B,) Board, one gather
        value, logits = model(boards)                                      # <- one batched call, all B nodes
        value, logits = value.float(), logits.float()                      # keep node stats fp32 even if the net runs in bf16
        legal = self.env.legal_action_mask(boards)
        P = t.softmax(t.where(legal, logits, -t.inf), dim=-1)
        value, legal, P = value.cpu(), legal.cpu(), P.cpu()
        values = []
        for b, node in enumerate(nodes):
            if node.is_terminal:
                values.append(node.terminal_value)    # value known at creation; net output discarded
            else:
                node.legal = legal[b]
                node.P = P[b]
                values.append(float(value[b]))
        return values

    @t.no_grad()
    def search(self, model, root_boards: Board, add_noise: bool = False) -> Float[Tensor, "B 7"]:
        """Run `cfg.sims` simulations of root-parallel MCTS from the (B,) `root_boards`; return (B, 7)
        root visit counts. Same signature as `BatchedMCTS.search`, just sequential-over-trees (so much slower)."""
        cfg = self.cfg
        B, device = root_boards.shape[0], root_boards.obs.device
        batch_idx = t.arange(B, device=device)

        # Shared board pool: every node just stores a `slot` into this, so we never pack/unpack boards.
        # At most one node is added per simulation, so `cfg.sims + 1` slots per game (slot 0 = root)
        # can never overflow.
        pool = Board(t.zeros((B, cfg.sims + 1, *root_boards.obs.shape[1:]), device=device),
                     t.zeros((B, cfg.sims + 1), dtype=t.bool, device=device))
        pool[:, 0] = root_boards                     # drop in all root boards at once (slot 0)
        nptr = [1] * B                               # next free slot per game

        # construct the roots of the trees, each pointing at slot 0 of its game's pool
        roots = []
        for b in range(B):
            node = Node(board=None)
            node.slot = 0
            roots.append(node)

        # evaluate every root in one forward pass to set its P / legal (as section 2 does for its single root)
        self._evaluate_batch(model, pool, batch_idx, roots)
        if add_noise:  # batched Dirichlet noise on the root priors, exactly as `expand_root` does it
            P = dirichlet_root_noise(t.stack([r.P for r in roots]), t.stack([r.legal for r in roots]),
                                     cfg.dirichlet_alpha, cfg.dirichlet_eps)
            for b, root in enumerate(roots):
                root.P = P[b]

        for _ in range(cfg.sims):
            # 1. SELECT + EXPAND each tree sequentially, collecting one leaf per tree
            leaves = []
            for game, root in enumerate(roots):
                node, action = select(root, cfg.c_puct)
                leaf = node if node.is_terminal else self._expand(pool, nptr, game, node, action)
                leaves.append(leaf)
            # 2. EVALUATE every leaf in a single, fixed-size forward pass through the network
            values = self._evaluate_batch(model, pool, batch_idx, leaves)
            # 3. BACKUP each tree sequentially
            for leaf, value in zip(leaves, values):
                backup(leaf, value)
        return t.stack([root.N for root in roots]).to(device)      # (B, 7), on the input's device

tests.test_simulated_batched_mcts(SimulatedBatchedMCTS)
# %%
def compute_z_targets(
    dones: Bool[Tensor, "batch timesteps"], 
    rewards: Float[Tensor, "batch timesteps"]
) -> Float[Tensor, "batch timesteps"]:
    """Negamax value targets for a batch of `B` self-play games of `T` plies.

    Walking each game backwards from its terminal rewards, the target at each ply is the game's final
    reward with its sign flipped once per step back. Every recorded value in this project is from the
    perspective of the player about to move; stepping back one ply changes whose turn it is, hence the
    negation (negamax: good for the mover is bad for its parent).

    Args:
        dones: (batch, timesteps) marks the ply where each game ended
        rewards:  (batch, timesteps) rewards to the mover at each ply (nonzero only where dones)

    Returns:
        (batch, timesteps) the mover-perspective outcome `z` for every recorded state
    """
    batch, timesteps = dones.shape
    z = t.zeros((batch, timesteps), device=dones.device)
    running = t.zeros((batch,), device=dones.device)
    for step in range(timesteps - 1, -1, -1):
        running = t.where(dones[:, step], rewards[:, step], -running)
        z[:, step] = running
    return z

def compute_z_targets_2(dones, rewards):
    B, T = dones.shape
    steps = t.arange(T, device=dones.device)                     # (T,)
    # each done ply's own index; T as a "no done here" sentinel
    done_idx = t.where(dones, steps, T)                          # (B, T)
    # next done at or after each ply = reverse cumulative min
    end = done_idx.flip(-1).cummin(-1).values.flip(-1)           # (B, T), values in [t, T]
    # reward at that done; the padded zero column makes the sentinel gather 0
    r_end = F.pad(rewards, (0, 1)).gather(-1, end)               # (B, T)
    # one sign flip per ply between t and its game's end
    sign = 1 - 2 * ((end - steps) % 2)
    return sign * r_end


tests.test_compute_z_targets(compute_z_targets)
# %%
# %%
from torch.utils import benchmark

def make_z_inputs(B=8192, T=42, device=device, seed=0):
    """Self-play-shaped (dones, rewards) like one generation from AZConfig's defaults (8192 games × 42 plies).
    Each row is back-to-back games starting at ply 0 (self_play resets every generation), with game
    lengths uniform in [7, 42] plies (7 = the shortest possible Connect 4 game). The last game in a row
    is usually cut off by the end of the generation, as in real self-play."""
    g = t.Generator().manual_seed(seed)
    lengths = t.randint(7, T + 1, (B, T // 7 + 1), generator=g)   # 7 games of >= 7 plies always overrun T
    ends = (lengths.cumsum(-1) - 1).clamp(max=T)                   # final ply of each game; T = past the end
    d = t.zeros(B, T + 1).scatter_(1, ends, 1.0)[:, :T]            # extra column soaks up the overruns
    dones = d.bool().to(device)
    rewards = dones.float()   # +1 to the mover at every game end (draws are rare enough to ignore here)
    return dones, rewards

devices = [t.device("cpu")] + ([device] if device.type != "cpu" else [])
for dev in devices:
    dones, rewards = make_z_inputs(device=dev)
    # correctness check first; doubles as warm-up (CUDA init, kernel caching)
    t.testing.assert_close(compute_z_targets(dones, rewards), compute_z_targets_2(dones, rewards))
    print(f"{dev}: {dones.shape[0]} rows × {dones.shape[1]} plies, {int(dones.sum())} finished games")
    medians = {}
    for fn in (compute_z_targets, compute_z_targets_2):
        m = benchmark.Timer(
            stmt="fn(dones, rewards)",
            globals={"fn": fn, "dones": dones, "rewards": rewards},
            num_threads=t.get_num_threads(),   # Timer defaults to 1 thread; match the notebook's setting
        ).blocked_autorange(min_run_time=1.0)
        medians[fn.__name__] = m.median
        print(f"  {fn.__name__:<22} median {m.median * 1e6:9.1f} µs   (IQR {m.iqr * 1e6:.1f} µs)")
    print(f"  speedup: {medians['compute_z_targets'] / medians['compute_z_targets_2']:.1f}×\n")
# %%
def compute_az_loss(
    value: Float[Tensor, "N"],
    logits: Float[Tensor, "N 7"],
    pi: Float[Tensor, "N 7"],
    z: Float[Tensor, "N"],
    value_coef: float = 1.0,
) -> Float[Tensor, ""]:
    """Scalar AlphaZero loss over a minibatch of `N` positions: policy cross-entropy + value MSE.

    Loss = mean of `-sum_a pi_a log softmax(logits)_a` + `value_coef * (value - z)^2`.

    Args:
        value:      (N,) critic outputs
        logits:     (N, 7) actor outputs
        pi:         (N, 7) MCTS visit-count policy target
        z:          (N,) game-outcome value target
        value_coef: weight on the value-MSE term

    Returns:
        scalar tensor: the mean total loss
    """
    assert value.shape == z.shape
    assert logits.shape == pi.shape
    policy_loss = (-logits.log_softmax(-1) * pi).sum(-1)
    value_loss = value_coef * (value - z).pow(2)
    loss = (policy_loss + value_loss).mean()
    return loss


tests.test_compute_az_loss(compute_az_loss)
# %%
@dataclass
class AZConfig:
    """All the knobs for self-play + training. The defaults are a fast in-notebook recipe (8192
    self-play games/gen, 16 sims/move, 8 generations, minibatch 2048) that reaches ~85% Pons-solver
    accuracy in ~3-4 min on a GPU (~2 min on an RTX 5090). For a stronger agent raise `sims` to 64 and `num_generations` to ~50
    (≈25k optimiser steps); to run faster still, dial `num_games` / `sims` / `num_generations` down.

    c_puct (1.5->1.0), lr (1e-3->5e-3) and buffer_gens (8->4) were tuned by a
    full-run sweep (pons_CE ~0.466->0.420 at ~1/4 the compute); every gain came from fresher self-play
    data — capacity/loss/regularisation were inert. Usable lr band ~3e-3..6e-3 (≥7e-3 is seed-unstable)."""
    # self-play / data
    num_games: int = 8192          # parallel self-play games per generation (4096 -> 8192: ~1.5x the games/sec, fewer gens needed)
    sims: int = 16                 # MCTS simulations per move (16 = the fast default; 32 learns about as well as 64 at ~2x the self-play speed, and 64 gives the strongest run)
    num_generations: int = 8        # training generations (~85% Pons acc in ~3-4 min; raise to ~50 for the full recipe)
    buffer_gens: int = 4           # replay buffer = the last this-many generations (tuned 8->4: fresher data)
    moves_per_gen: int = 42        # plies per generation (a full Connect-4 game)
    temperature: float = 1.0       # visit-count sampling temperature, used for every ply
    # MCTS / exploration
    c_puct: float = 1.0               # tuned 1.5->1.0 (full-run sweep, both seeds)
    max_depth: int = 42
    dirichlet_alpha: float = 10 / 7   # ≈ 1.43, root exploration-noise concentration
    dirichlet_eps: float = 0.25       # weight of the root Dirichlet noise
    # optimiser / schedule
    lr: float = 5e-3               # initial learning rate (tuned 1e-3->5e-3; >=7e-3 is seed-unstable)
    lr_min: float = 2e-5           # cosine-decay target over the run
    weight_decay: float = 1e-4
    grad_clip: float = 1.0         # global grad-norm clip
    minibatch: int = 2048          # 1024 -> 2048: ~2x faster training pass (1024 leaves the GPU ~25% busy), same learning
    value_coef: float = 1.0        # weight on the value-MSE loss term
    # precision: bf16 autocast on by default if supported
    use_bf16: bool = device.type == "cuda" and t.cuda.is_bf16_supported(including_emulation=False)
    # logging
    use_wandb: bool = False        # log loss / lr / Pons metrics to Weights & Biases
    wandb_project: str = "alphazero-connect4"
# %%
class ReplayBuffer:
    """Given. Rolling replay of the last `cfg.buffer_gens` self-play generations.

    Usage: `write(...)` one ply at a time, `end_generation()` after each generation (computes value
    targets, drops unfinished states, flattens over (game, ply), evicts the oldest generation), then
    `minibatches(mb)` for a shuffled list of training minibatches. Mirrors the VPG rollout buffer from [2.2]."""

    def __init__(self, cfg: AZConfig, device):
        self.cfg, self.device = cfg, device
        B, T = cfg.num_games, cfg.moves_per_gen
        # preallocated rollout for the CURRENT generation (written one ply at a time, in place)
        self.boards = Board(t.empty((B, T, 3, 6, 7), device=device),
                            t.empty((B, T), dtype=t.bool, device=device))
        self.pi = t.empty((B, T, 7), device=device)
        self.dones = t.empty((B, T), dtype=t.bool, device=device)
        self.rews = t.empty((B, T), device=device)
        self.t = 0                 # next free ply slot in the current generation
        self.gens = []             # rolling list of finished generations, each a flat (boards, pi, z)

    def write(self, boards, pi, done, reward):
        """Record one ply of the current generation into row `self.t`, then advance."""
        self.boards[:, self.t] = boards
        self.pi[:, self.t] = pi
        self.dones[:, self.t] = done
        self.rews[:, self.t] = reward
        self.t += 1

    def end_generation(self):
        """Finish the current generation: compute negamax `z`, keep only states whose game finished
        (reverse cumulative-OR of dones over time), flatten over (game, ply), append to the rolling
        buffer (evicting the oldest), and reset the write pointer."""
        z = compute_z_targets(self.dones, self.rews)                          # (B, T)
        keep = (self.dones.int().flip(-1).cumsum(-1).flip(-1) > 0).reshape(-1)
        self.gens.append((self.boards.reshape(-1)[keep],
                          self.pi.reshape(-1, 7)[keep],
                          z.reshape(-1)[keep]))
        if len(self.gens) > self.cfg.buffer_gens:
            self.gens.pop(0)
        self.t = 0

    def minibatches(self, batch_size):
        """Snapshot the whole buffer into a freshly shuffled list of `(boards, pi, z)` minibatches
        (one epoch). The last, partial batch is dropped so every batch is exactly `batch_size` (great
        for a fixed compiled forward), but only when there's at least one full batch, so tiny configs
        still train."""
        boards = Board.cat([g[0] for g in self.gens])
        pi = t.cat([g[1] for g in self.gens])
        z = t.cat([g[2] for g in self.gens])
        perm = t.randperm(len(z), device=z.device)
        if len(perm) >= batch_size:
            perm = perm[: len(perm) // batch_size * batch_size]   # drop the last partial batch
        return [(boards[idx], pi[idx], z[idx]) for idx in perm.split(batch_size)]

    def reset(self):
        self.gens, self.t = [], 0

    def __len__(self):
        return sum(len(g[2]) for g in self.gens)
# %%
from pascal_pons.eval_pons import evaluate_policy
# %%
class AlphaZeroTrainer:
    """Owns the full per-generation loop described below. Each generation `train()` runs:

    1. `self_play()`: `moves_per_gen` plies of batched self-play with the frozen net
       (your `self_play_step` per ply, writing `(boards, π, done, reward)` into `self.buffer`),
       then `buffer.end_generation()` turns the rollout into flat `(boards, π, z)` training rows.
    2. one supervised pass over `buffer.minibatches(minibatch)` (your `training_step` per batch),
    3. a cosine LR-schedule step and (periodically) `evaluate()` against the Pons solver.

    Attributes: `env` (Connect4Env), `cfg` (AZConfig), `model` (Connect4Model), `opt` (AdamW),
    `mcts` (BatchedMCTS — built from cfg's sims/c_puct/dirichlet settings), `buffer` (ReplayBuffer).
    """
    def __init__(self, env, cfg, model):
        self.env = env
        self.cfg = cfg
        self.device = env.device
        self.model = model
        self.opt = t.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.mcts = BatchedMCTS(env, MCTSConfig(
            sims=cfg.sims, c_puct=cfg.c_puct, max_depth=cfg.max_depth,
            dirichlet_alpha=cfg.dirichlet_alpha, dirichlet_eps=cfg.dirichlet_eps))
        self.buffer = ReplayBuffer(cfg, self.device)


    def visit_policy(self, root_N: Float[Tensor, "B 7"], temperature: float = 1.0) -> Float[Tensor, "B 7"]:
        """The tree policy π(a) ∝ N(s,a)^(1/τ), per game. At τ = 1 this is the training target; with
        `cfg.temperature` it is the distribution the played move is sampled from.

        Args:
            root_N:      (B, 7) root visit counts from the MCTS search
            temperature: τ; 1.0 is proportional to visit counts, →0 approaches greedy argmax,
                         larger flattens the distribution

        Returns:
            (B, 7) probabilities, each row summing to 1. Actions with zero visits get probability 0.
        """
        raise NotImplementedError()

    @t.no_grad()
    def self_play_step(self, boards: Board) -> tuple[Board, Bool[Tensor, "B"]]:
        """One ply of self-play for all `num_games` games at once: MCTS -> tree policy -> sample ->
        step -> record the ply into `self.buffer`.

        Args:
            boards: (B,) current boards

        Returns:
            next_boards: (B,) boards after the move (finished games auto-reset to an empty board)
            done:        (B,) whether the move ended each game
        """
        raise NotImplementedError()

    @t.no_grad()
    def self_play(self, progress: bool = True):
        """Play one generation: `num_games` games for `moves_per_gen` plies, calling your
        `self_play_step` each ply (which `write`s into `self.buffer`), then finalise the generation in
        the buffer. The buffer handles the value targets, masking and replay -- nothing to return.

        The ply loop is wrapped in a tqdm bar with `unit_scale` (like the 2.2 SPS bar) so you get a
        live, SI-formatted env-steps/sec readout during self-play rather than one update per generation."""
        B, T = self.cfg.num_games, self.cfg.moves_per_gen
        boards = self.env.reset(B)
        self.model.eval()
        # each ply runs `sims` batched env.steps over all B games (one per MCTS simulation), so the
        # generation does B * T * sims env transitions in total -- tqdm turns that into a live rate,
        # and the postfix shows cumulative MCTS simulations done out of `moves_per_gen * sims`.
        total_sims = T * self.cfg.sims
        bar = tqdm(total=B * T * self.cfg.sims, unit=" env steps", unit_scale=True,
                   desc="self-play", leave=False, disable=not progress)
        
        with t.autocast(self.device.type, dtype=t.bfloat16, enabled=self.cfg.use_bf16):
            for ply in range(T):
                boards, _ = self.self_play_step(boards)   # finished games auto-reset (red to move)
                bar.update(B * self.cfg.sims)
                bar.set_postfix_str(f"sims {(ply + 1) * self.cfg.sims}/{total_sims}")
        bar.close()
        self.buffer.end_generation()

    def training_step(self, boards, pi, z):
        """One optimiser step on a single minibatch `(boards, pi, z)`: forward the net, compute the
        AlphaZero loss (`compute_az_loss`), zero the grads (`self.opt.zero_grad(set_to_none=True)`),
        backprop, clip the gradient norm to `cfg.grad_clip`, and step the optimiser. AlphaZero's update is just supervised learning -- regress the value head onto
        `z` and the policy head onto `pi`.

        Args:
            boards: (mb,) Board minibatch
            pi:  (mb, 7) MCTS policy targets
            z:   (mb,) value targets

        Returns:
            float: the minibatch loss
        """
        raise NotImplementedError()

    @t.no_grad()
    def evaluate(self) -> dict:
        """Given. Score the current network against the frozen Pons solver set (policy accuracy /
        cross-entropy / value sign-accuracy). One cached forward pass; see the section above."""
        return evaluate_policy(self.model, self.env)

    def train(self, num_generations=None, eval_every=1):
        """Given. The full training loop -- you don't need to touch this. Each generation: run
        self-play into the buffer, do one supervised pass over the buffer's minibatches
        (calling your `training_step`), step the cosine LR schedule, and periodically `evaluate`
        against the Pons solver. Logs loss / lr / eval to a tqdm bar (and to wandb if `cfg.use_wandb`)."""
        import time
        num_generations = num_generations or self.cfg.num_generations
        # cosine-decay the LR; schedule over a >=10-gen horizon so a short quick run doesn't crater the
        # LR before it finishes (a 6-gen T_max=6 cosine would decay to lr_min by the last gen and stall).
        sched = t.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=max(num_generations, 10), eta_min=self.cfg.lr_min)
        # effective env transitions/sec for the tqdm bar: each generation steps the env
        # num_games * moves_per_gen * sims times (one batched env.step per MCTS simulation). It's just a
        # counter + a clock, so it adds no per-step overhead -- handy for spotting when self-play bottlenecks.
        t0, env_steps = time.time(), 0
        steps_per_gen = self.cfg.num_games * self.cfg.moves_per_gen * self.cfg.sims
        if self.cfg.use_wandb:
            import wandb
            wandb.init(project=self.cfg.wandb_project, config=asdict(self.cfg))
        metrics = {}
        bar = tqdm(range(1, num_generations + 1))
        for gen in bar:
            self.self_play()                                       # fill + roll the replay buffer
            env_steps += steps_per_gen
            self.model.train()
            batches = self.buffer.minibatches(self.cfg.minibatch)
            total_loss, n_batches = 0.0, 0
            tbar = tqdm(batches, desc="train", leave=False)       # one supervised pass over the buffer
            for boards, pi, z in tbar:
                with t.autocast(self.device.type, dtype=t.bfloat16, enabled=self.cfg.use_bf16):
                    total_loss += self.training_step(boards, pi, z)
                n_batches += 1
                tbar.set_postfix_str(f"loss={total_loss / n_batches:.3f}")   # live running-mean loss
            loss = total_loss / max(n_batches, 1)
            sched.step()
            if eval_every and gen % eval_every == 0:
                metrics = self.evaluate()                          # Pons solver benchmark
            lr = sched.get_last_lr()[0]
            sps = env_steps / max(time.time() - t0, 1e-9)          # effective env steps/sec (cumulative)
            bar.set_postfix_str(f"loss={loss:.3f}  acc={metrics.get('pons/acc', float('nan')):.3f}  "
                                f"ce={metrics.get('pons/ce', float('nan')):.3f}  env/s={fmt_si(sps)}")
            if self.cfg.use_wandb:
                wandb.log({"generation": gen, "loss": loss, "lr": lr, "env_steps_per_sec": sps, **metrics})
        if self.cfg.use_wandb:
            wandb.finish()
        return self.model


tests.test_visit_policy(AlphaZeroTrainer)
tests.test_self_play_step(AlphaZeroTrainer)
tests.test_training_step(AlphaZeroTrainer)