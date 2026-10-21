"""
MCTS for chess. The same algorithm as the Connect-4 single-game search (select -> expand -> evaluate
-> backup, PUCT with Q = W / max(N, 1), negamax backup), run over many trees at once with ONE batched
network call per simulation -- i.e. the Connect-4 `SimulatedBatchedMCTS` design.

Differences from Connect-4:
  * per-node edge statistics are numpy arrays over the node's LEGAL moves only (~35 entries, not 4672);
    `node.lidx` maps each legal move to its network action index.
  * the network is reached through an `evaluator(obs_u8, lidx) -> (priors, values)` callable, so the
    same search runs in-process (`LocalEvaluator`) or in a self-play worker talking to the GPU
    inference server (`selfplay.RemoteEvaluator`).
  * trees can be reused between moves (`advance`): the chosen child becomes the next root.
"""
from __future__ import annotations

import math

import chess
import numpy as np
import torch

from chapter2_rl.exercises.part5_alphazero_chess.chess_env import N_IN_PLANES, encode_board, legal_actions, terminal_value
from chapter2_rl.exercises.part5_alphazero_chess.model import infer


class Node:
    """One board state. Edge statistics (P, N, W, children) are indexed by position in `self.moves`."""
    __slots__ = ("board", "moves", "lidx", "P", "P_raw", "N", "W", "n_total", "children", "parent",
                 "pidx", "terminal", "tval", "value", "Qf", "U", "fpu")

    def __init__(self, board: chess.Board, parent: "Node | None" = None, pidx: int = -1):
        self.board = board
        self.parent = parent
        self.pidx = pidx                       # index of the edge (in parent.moves) that led here
        self.moves = list(board.legal_moves)
        self.tval = terminal_value(board, len(self.moves))
        self.terminal = self.tval is not None
        self.P = None                          # prior over self.moves, set when evaluated
        self.P_raw = None                      # pre-noise prior (only set on noised roots)
        self.value = None                      # network value for the side to move, set when evaluated
        L = len(self.moves)
        self.lidx = legal_actions(board, self.moves) if not self.terminal else []
        self.N = np.zeros(L, dtype=np.float32)
        self.W = np.zeros(L, dtype=np.float32)
        self.n_total = 0
        self.children: list[Node | None] = [None] * L
        self.Qf = self.U = self.fpu = None       # PUCT caches, see select_edge

    @property
    def Q(self) -> np.ndarray:
        return self.W / np.maximum(self.N, 1.0)

    @property
    def evaluated(self) -> bool:
        return self.P is not None

    def child(self, i: int) -> "Node":
        """The child along edge i, created (board copy + push) on first use."""
        c = self.children[i]
        if c is None:
            b = self.board.copy(stack=8)           # 8 plies of history is enough for threefold detection
            b.push(self.moves[i])
            c = self.children[i] = Node(b, self, i)
        return c


# --------------------------------------------------------------------------- evaluation
class LocalEvaluator:
    """In-process network evaluator: numpy (obs, lidx) -> numpy (priors, values)."""
    def __init__(self, model: torch.nn.Module, device=None, max_batch: int = 1024):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.max_batch = max_batch

    def __call__(self, obs: np.ndarray, lidx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        ps, vs = [], []
        for s in range(0, len(obs), self.max_batch):
            o = torch.from_numpy(obs[s:s + self.max_batch]).to(self.device)
            li = torch.from_numpy(lidx[s:s + self.max_batch]).to(self.device).long()
            v, p = infer(self.model, o, li)
            ps.append(p.cpu().numpy()); vs.append(v.cpu().numpy())
        return np.concatenate(ps), np.concatenate(vs)


class UniformEvaluator:
    """Uniform prior + zero value: tests the search logic in isolation (cf. the Connect-4 `DummyModel`)."""
    def __call__(self, obs, lidx):
        legal = (lidx >= 0).astype(np.float32)
        return legal / legal.sum(1, keepdims=True), np.zeros(len(obs), dtype=np.float32)


def encode_nodes(nodes: list[Node]) -> tuple[np.ndarray, np.ndarray]:
    """Network inputs for `nodes`: uint8 planes (n, 19, 8, 8) and legal action indices (n, L), -1 padded."""
    L = max(len(n.moves) for n in nodes)
    obs = np.empty((len(nodes), N_IN_PLANES, 8, 8), dtype=np.uint8)
    lidx = np.full((len(nodes), L), -1, dtype=np.int16)
    for i, n in enumerate(nodes):
        obs[i] = encode_board(n.board)
        lidx[i, :len(n.lidx)] = n.lidx
    return obs, lidx


def apply_evaluations(nodes: list[Node], priors: np.ndarray, values: np.ndarray) -> None:
    for i, n in enumerate(nodes):
        n.P = np.array(priors[i, :len(n.moves)], dtype=np.float32)   # copy: the evaluator may reuse its buffers
        n.value = float(values[i])


def evaluate_nodes(nodes: list[Node], evaluator) -> None:
    """One batched network call for all `nodes` (non-terminal, unevaluated); sets node.P / node.value."""
    if nodes:
        apply_evaluations(nodes, *evaluator(*encode_nodes(nodes)))


def add_root_noise(root: Node, alpha: float, eps: float, rng: np.random.Generator) -> None:
    """Mix Dirichlet noise into an evaluated root's prior (the raw prior is kept in `P_raw`)."""
    if root.P_raw is None:
        root.P_raw = root.P
    root.P = (1 - eps) * root.P_raw + eps * rng.dirichlet(np.full(len(root.moves), alpha)).astype(np.float32)
    root.Qf = None                                 # P changed: rebuild the PUCT caches


# --------------------------------------------------------------------------- search
def select_edge(node: Node, c_puct: float, fpu_reduction: float | None = 0.0) -> int:
    """PUCT-best edge: argmax Q + c * P * sqrt(1 + sum N) / (1 + N), over the legal moves.

    First-play urgency: an unvisited edge's Q is taken to be the node's own network value minus
    `fpu_reduction` (Leela Chess Zero style) rather than 0 as in the Connect-4 code. With Q = 0, in a
    position the network thinks is bad every visited move looks worse than every unvisited one, so the
    search smears its visits over all ~35 moves. `fpu_reduction=None` restores the Connect-4 rule.

    Speed: this runs once per tree level per simulation, so Q (FPU-filled) and P / (1 + N) are cached
    on the node (`Qf`, `U`) and updated edge-by-edge in `backup`, leaving one fused expression here.
    """
    if node.Qf is None or node.fpu != fpu_reduction:
        _init_puct_cache(node, fpu_reduction)
    return int(np.argmax(node.Qf + (c_puct * math.sqrt(node.n_total + 1)) * node.U))


def _init_puct_cache(node: Node, fpu_reduction: float | None) -> None:
    q = node.W / np.maximum(node.N, 1.0)
    if fpu_reduction is not None:
        q = np.where(node.N > 0, q, np.float32(node.value - fpu_reduction))
    node.Qf, node.U, node.fpu = q.astype(np.float32), node.P / (1.0 + node.N), fpu_reduction


def backup(leaf: Node, value: float) -> None:
    """Negamax backup: walk to the root, flipping the value's sign at every edge."""
    node, v = leaf, value
    while node.parent is not None:
        v = -v
        p, i = node.parent, node.pidx
        n = p.N[i] + 1.0
        w = p.W[i] + v
        p.N[i], p.W[i] = n, w
        p.n_total += 1
        if p.Qf is not None:                       # keep the select_edge caches in step
            p.Qf[i] = w / n
            p.U[i] = p.P[i] / (1.0 + n)
        node = p


def select_leaves(roots: list[Node], c_puct: float, fpu_reduction: float | None = 0.0) -> tuple[list[Node], list[Node]]:
    """SELECT + EXPAND for one simulation of every tree: walk down by PUCT to a new (or terminal) node.
    Returns (leaves, fresh): one leaf per root, and the subset that are new and need a network evaluation."""
    leaves, fresh = [], []
    for root in roots:
        node = root
        while True:
            child = node.child(select_edge(node, c_puct, fpu_reduction))
            if child.terminal or not child.evaluated:
                break
            node = child
        leaves.append(child)
        if not child.terminal:
            fresh.append(child)
    return leaves, fresh


def backup_leaves(leaves: list[Node]) -> None:
    for leaf in leaves:
        backup(leaf, leaf.tval if leaf.terminal else leaf.value)


def search(roots: list[Node], evaluator, sims: int, c_puct: float = 2.0, fpu_reduction: float | None = 0.0) -> None:
    """Run `sims` simulations on every (non-terminal) root, one batched network call per simulation.

    Each simulation walks every tree down to a new (or terminal) node, evaluates all the new nodes in
    a single call, then backs each leaf's value up its own tree. Roots must already be evaluated
    (`prepare_roots`). Statistics accumulate in place, so call again to search deeper.
    """
    roots = [r for r in roots if not r.terminal]
    for _ in range(sims):
        leaves, fresh = select_leaves(roots, c_puct, fpu_reduction)
        evaluate_nodes(fresh, evaluator)
        backup_leaves(leaves)


def prepare_roots(roots: list[Node], evaluator, noise: tuple[float, float] | None = None,
                  rng: np.random.Generator | None = None) -> None:
    """Evaluate any unevaluated roots (one batched call) and optionally add Dirichlet (alpha, eps) noise."""
    evaluate_nodes([r for r in roots if not r.terminal and not r.evaluated], evaluator)
    if noise is not None:
        rng = rng or np.random.default_rng()
        for r in roots:
            if not r.terminal:
                add_root_noise(r, *noise, rng)


def visit_policy(root: Node) -> np.ndarray:
    """The MCTS policy target pi(a) = N(s, a) / sum_a' N(s, a') over root.moves."""
    return root.N / max(root.N.sum(), 1.0)


def choose_move(root: Node, temperature: float, rng: np.random.Generator | None = None) -> int:
    """Pick an edge from root visit counts: argmax when temperature ~ 0, else sample N^(1/T)."""
    if temperature < 1e-3:
        best = np.flatnonzero(root.N == root.N.max())
        return int(best[np.argmax(root.P[best])]) if len(best) > 1 else int(best[0])   # tie-break on prior
    logits = np.log(np.maximum(root.N, 1e-12)) / temperature
    p = np.exp(logits - logits.max())
    p /= p.sum()
    rng = rng or np.random.default_rng()
    return int(rng.choice(len(p), p=p))


def advance(root: Node, move: chess.Move) -> Node:
    """Tree reuse: return the child for `move` as a detached new root (created if unexplored)."""
    i = root.moves.index(move)
    child = root.child(i)
    child.parent = None
    if child.P_raw is not None:                    # never happens for a non-root, but keep roots clean
        child.P, child.P_raw, child.Qf = child.P_raw, None, None
    return child


def new_root(board: chess.Board | None = None) -> Node:
    return Node(board.copy(stack=8) if board is not None else chess.Board())


# --------------------------------------------------------------------------- one-position analysis
def analyse(board: chess.Board, evaluator, sims: int = 200, c_puct: float = 2.0, root: Node | None = None,
            fpu_reduction: float | None = 0.0) -> dict:
    """Search one position and summarise it: moves sorted by visits, with prior, visit share and Q.

    Returns dict(root, value (network value for the side to move), moves [{move, san, uci, visits,
    share, prior, q}], best (chess.Move)).
    """
    root = root or new_root(board)
    if root.terminal:
        return {"root": root, "value": root.tval, "moves": [], "best": None}
    prepare_roots([root], evaluator)
    search([root], evaluator, sims, c_puct, fpu_reduction)
    order = np.argsort(-(root.N + 1e-3 * root.P))
    share = visit_policy(root)
    moves = [{"move": root.moves[i], "san": root.board.san(root.moves[i]), "uci": root.moves[i].uci(),
              "visits": int(root.N[i]), "share": float(share[i]), "prior": float(root.P[i]),
              "q": float(root.Q[i])} for i in order]
    return {"root": root, "value": root.value, "moves": moves, "best": moves[0]["move"]}
