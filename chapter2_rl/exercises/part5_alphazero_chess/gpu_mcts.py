"""
Fully vectorised MCTS for chess on the GPU: the chess version of the Connect-4 `BatchedMCTS` / `Tree`.

B games are searched in lock-step. Every tree lives in flat (B, nodes, ...) tensors -- node 0 is the root,
one node is added per simulation, and an extra "dustbin" row absorbs the writes of games that are not
expanding -- so a simulation is a fixed sequence of batched tensor ops with no per-game Python:

    SELECT    walk every tree down by PUCT (one batched step per depth) to an unexpanded edge or a terminal
    EXPAND    one `torch_chess.step` for all B parent states along the chosen edges
    EVALUATE  one network call on the B new positions (CUDA-graph replay)
    BACKUP    negamax up the parent pointers

Unlike the Connect-4 tree, edges are stored per *legal move* rather than per action (a node has <= 218
legal moves but 4672 actions), so per-edge tensors are (B, nodes, 256) and `lidx` maps an edge to its
action. PUCT matches `mcts.select_edge` (including first-play urgency from the node's own value), so for
a deterministic network this search gives the same visit counts as the CPU `mcts.search` without reuse.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

import chapter2_rl.exercises.part5_alphazero_chess.torch_chess as TC
from chapter2_rl.exercises.part5_alphazero_chess.chess_env import MAX_LEGAL

KL = MAX_LEGAL


@dataclass
class SearchConfig:
    sims: int = 64
    c_puct: float = 2.0
    fpu_reduction: float = 0.0
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25


class GPUTree:
    """Node pools for B trees of up to `sims + 1` nodes each (+1 dustbin row)."""

    STATE_FIELDS = ("board", "white", "castling", "ep", "halfmove", "ply", "hist")

    def __init__(self, B: int, sims: int, device):
        M = sims + 2                                    # root + one node per simulation (+ spare)
        self.B, self.M, self.DUST, self.dev = B, M, M, device
        z = lambda *shape, dtype=torch.float32, fill=0: torch.full((B, M + 1, *shape), fill, dtype=dtype, device=device)
        self.board = z(64, dtype=torch.int8)
        self.white = z(dtype=torch.bool)
        self.castling = z(4, dtype=torch.bool)
        self.ep = z(dtype=torch.int64, fill=-1)
        self.halfmove = z(dtype=torch.int64)
        self.ply = z(dtype=torch.int64)
        self.hist = z(TC.HIST, dtype=torch.int64)
        self.terminal = z(dtype=torch.bool)
        self.tval = z()
        self.value = z()
        self.lidx = z(KL, dtype=torch.int16, fill=-1)
        self.nleg = z(dtype=torch.int64)
        self.P = z(KL)
        self.N = z(KL)
        self.W = z(KL)
        self.child = z(KL, dtype=torch.int32, fill=-1)
        self.parent = z(dtype=torch.int64, fill=-1)
        self.pedge = z(dtype=torch.int64)
        self.nptr = torch.ones(B, dtype=torch.int64, device=device)
        self.ar = torch.arange(B, device=device)

    def reset(self) -> None:
        """Empty the trees so the pools can be reused for the next move (only the statistics, the links and
        the node pointer need clearing: every other field is written when its node is created)."""
        self.N.zero_()
        self.W.zero_()
        self.child.fill_(-1)
        self.parent.fill_(-1)
        self.nptr.fill_(1)

    def get_state(self, node: torch.Tensor) -> TC.ChessState:
        ar = self.ar
        return TC.ChessState(**{f: getattr(self, f)[ar, node] for f in self.STATE_FIELDS})

    def put_state(self, slot: torch.Tensor, s: TC.ChessState) -> None:
        """Write a finalized state into node `slot` of each game, plus its legal-move list."""
        ar = self.ar
        for f in self.STATE_FIELDS:
            getattr(self, f)[ar, slot] = getattr(s, f)
        lidx, n = TC.compact_legal(s.legal, KL)
        self.lidx[ar, slot] = lidx.to(torch.int16)
        self.nleg[ar, slot] = n
        self.terminal[ar, slot] = s.terminal
        self.tval[ar, slot] = s.value


def _evaluate(net, s: TC.ChessState, lidx: torch.Tensor):
    """(values (B,), priors (B, KL)) for states `s` whose legal moves are `lidx` (B, KL) int (-1 padded)."""
    return net.run(TC.encode(s), lidx.long())


def _dirichlet(nleg: torch.Tensor, alpha: float) -> torch.Tensor:
    """Dirichlet(alpha) noise over each row's first nleg entries (normalised gammas), (B, KL)."""
    g = torch.distributions.Gamma(torch.full((len(nleg), KL), alpha, device=nleg.device), 1.0).sample()
    g = g * (torch.arange(KL, device=nleg.device) < nleg.unsqueeze(1))
    return g / g.sum(1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def search(net, root: TC.ChessState, cfg: SearchConfig, add_noise: bool = True, tree: GPUTree | None = None):
    """Search B (non-terminal) root positions. `net` is a `model.GraphedInference` (or anything with
    `.run(obs, lidx)`). Returns (visits (B, KL), lidx (B, KL) long actions of the root edges, root value (B,))."""
    B, dev = len(root), root.board.device
    if tree is None:
        T = GPUTree(B, cfg.sims, dev)
    else:
        T = tree
        assert T.B == B and T.M >= cfg.sims + 2, "tree too small for this search"
        T.reset()
    ar = T.ar
    zero = torch.zeros(B, dtype=torch.int64, device=dev)
    T.put_state(zero, root)
    v, p = _evaluate(net, root, T.lidx[:, 0])
    if add_noise:
        p = (1 - cfg.dirichlet_eps) * p + cfg.dirichlet_eps * _dirichlet(T.nleg[:, 0], cfg.dirichlet_alpha)
    T.P[:, 0], T.value[:, 0] = p, v
    edge = torch.arange(KL, device=dev)
    maxd = min(cfg.sims + 1, 512)
    for _ in range(cfg.sims):
        # ---- SELECT: lock-step PUCT descent
        node = zero.clone()
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        leaf_is_term = torch.zeros_like(done)
        term_node = zero.clone()
        leaf_parent, leaf_edge = zero.clone(), zero.clone()
        expand = torch.zeros_like(done)
        for d in range(maxd):
            Nn, Wn, Pn = T.N[ar, node], T.W[ar, node], T.P[ar, node]
            q = torch.where(Nn > 0, Wn / Nn.clamp_min(1.0), (T.value[ar, node] - cfg.fpu_reduction).unsqueeze(1))
            score = q + (cfg.c_puct * torch.sqrt(Nn.sum(1, keepdim=True) + 1.0)) * Pn / (1.0 + Nn)
            score = score.masked_fill(edge >= T.nleg[ar, node].unsqueeze(1), float("-inf"))
            a = score.argmax(1)
            ch = T.child[ar, node, a].long()
            active = ~done
            is_term = active & T.terminal[ar, node]
            leaf_is_term |= is_term
            term_node = torch.where(is_term, node, term_node)
            walk = active & ~is_term
            new = walk & (ch < 0)
            leaf_parent = torch.where(new, node, leaf_parent)
            leaf_edge = torch.where(new, a, leaf_edge)
            expand |= new
            done |= is_term | new
            node = torch.where(walk & ~new, ch, node)
            if d >= 1 and bool(done.all()):
                break
        # ---- EXPAND: one batched move for every game (non-expanding games write to the dustbin)
        action = T.lidx[ar, leaf_parent, leaf_edge].long().clamp(min=0)
        ns = TC.step(T.get_state(leaf_parent), action)
        slot = torch.where(expand, T.nptr, T.DUST)
        T.put_state(slot, ns)
        T.parent[ar, slot] = leaf_parent
        T.pedge[ar, slot] = leaf_edge
        T.child[ar, leaf_parent, leaf_edge] = torch.where(expand, T.nptr, T.child[ar, leaf_parent, leaf_edge].long()).int()
        new_ids = slot
        T.nptr = T.nptr + expand.long()
        # ---- EVALUATE the new positions
        v, p = _evaluate(net, ns, T.lidx[ar, slot])
        T.P[ar, slot], T.value[ar, slot] = p, v
        leaf_value = torch.where(leaf_is_term, T.tval[ar, term_node],
                                 torch.where(ns.terminal, ns.value, v))
        leaf_node = torch.where(expand, new_ids, term_node)
        live = expand | leaf_is_term
        # ---- BACKUP (negamax) along parent pointers
        n_, val = leaf_node, torch.where(live, leaf_value, 0.0)
        for _ in range(maxd):
            at_root = (n_ == 0) | ~live
            par = T.parent[ar, n_].clamp(min=0)
            pe = T.pedge[ar, n_]
            val = torch.where(at_root, val, -val)
            w = (~at_root).float()
            T.N[ar, par, pe] += w
            T.W[ar, par, pe] += val * w
            n_ = torch.where(at_root, n_, par)
            if bool(((n_ == 0) | ~live).all()):
                break
    return T.N[:, 0].clone(), T.lidx[:, 0].long(), T.value[:, 0].clone()
