"""Tests + benchmark for the GPU MCTS. Run: `python test_gpu_mcts.py [bench]`."""
import sys
import time

import chess
import numpy as np
import torch

import chess_env as E
import gpu_mcts as G
import mcts as M
import torch_chess as TC
from opponents import OPENINGS, opening_board

DEV = torch.device("cuda")
MATE_IN_ONE = "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"


class MaterialNet:
    """Uniform prior over legal moves, value = (my material - their material) / 40. Deterministic and
    identical on CPU / GPU, so the two searches can be compared."""
    W = torch.tensor([1, 3, 3, 5, 9, 0], dtype=torch.float32)

    def _value(self, obs):
        o = obs.float()
        w = self.W.to(o.device)
        return ((o[:, 0:6].sum((2, 3)) * w).sum(1) - (o[:, 6:12].sum((2, 3)) * w).sum(1)) / 40.0

    def run(self, obs, lidx):                                   # GPU-search interface
        legal = (lidx >= 0).float()
        return self._value(obs), legal / legal.sum(1, keepdim=True)

    def __call__(self, obs, lidx):                              # CPU mcts evaluator interface (numpy)
        legal = (lidx >= 0).astype(np.float32)
        return legal / legal.sum(1, keepdims=True), self._value(torch.from_numpy(obs)).numpy()


class ZeroNet(MaterialNet):
    def _value(self, obs):
        return torch.zeros(obs.shape[0], device=obs.device)


def test_basic_and_mate():
    cfg = G.SearchConfig(sims=200)
    boards = [chess.Board(MATE_IN_ONE), chess.Board(MATE_IN_ONE).mirror(), chess.Board()]
    s = TC.from_boards(boards, DEV)
    visits, lidx, _ = G.search(ZeroNet(), s, cfg, add_noise=False)
    assert torch.allclose(visits.sum(1), torch.full((3,), 200.0, device=DEV)), visits.sum(1)
    assert bool(((lidx < 0) <= (visits == 0)).all()), "visits on padding edges"
    for i, want in [(0, "d1d8"), (1, "d8d1")]:
        best = int(lidx[i, visits[i].argmax()])
        got = E.action_to_move(best, boards[i]).uci()
        assert got == want, f"board {i}: GPU MCTS chose {got}, want mate {want}"
    print("test_basic_and_mate passed")


def _cpu_moves_in_action_order():
    """Make the CPU search list each node's moves in action-index order -- the GPU tree's edge order -- so
    exact ties in PUCT are broken the same way by both argmaxes."""
    orig = M.Node.__init__

    def init(self, board, parent=None, pidx=-1):
        orig(self, board, parent, pidx)
        if not self.terminal:
            order = np.argsort(self.lidx)
            self.moves = [self.moves[i] for i in order]
            self.lidx = [self.lidx[i] for i in order]
    M.Node.__init__ = init
    return orig


def test_agrees_with_cpu_search(sims=96):
    orig = _cpu_moves_in_action_order()
    try:
        _agrees(sims)
    finally:
        M.Node.__init__ = orig


def _agrees(sims):
    names = list(OPENINGS)
    boards = [opening_board(o) for o in names] + [chess.Board("4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1")]
    cfg = G.SearchConfig(sims=sims, fpu_reduction=0.0)
    visits, lidx, _ = G.search(MaterialNet(), TC.from_boards(boards, DEV), cfg, add_noise=False)
    tvs, same_top = [], 0
    for i, b in enumerate(boards):
        r = M.analyse(b, MaterialNet(), sims=sims, c_puct=cfg.c_puct, fpu_reduction=0.0)
        cpu = {m["uci"]: m["visits"] for m in r["moves"]}
        gpu = {E.action_to_move(int(a), b).uci(): float(v) for a, v in zip(lidx[i].tolist(), visits[i].tolist()) if a >= 0}
        assert set(cpu) == set(gpu), "different legal move sets"
        tv = 0.5 * sum(abs(cpu[k] - gpu[k]) for k in cpu) / sims
        tvs.append(tv)
        same_top += max(cpu, key=cpu.get) == max(gpu, key=gpu.get)
    print(f"test_agrees_with_cpu_search: mean visit-distribution distance {np.mean(tvs):.3f} "
          f"(max {np.max(tvs):.3f}), same top move in {same_top}/{len(boards)}")
    assert np.mean(tvs) < 0.02 and same_top >= len(boards) - 1


def bench(B=2048, sims=64, channels=128, blocks=10):
    from model import ChessNet, GraphedInference
    torch.backends.cudnn.benchmark = True
    net = ChessNet(channels, blocks).to(DEV).to(memory_format=torch.channels_last).eval()
    g = GraphedInference(net, DEV, buckets=(B,))
    s = TC.initial_state(B, DEV)
    G.search(g, s, G.SearchConfig(sims=4))
    torch.cuda.synchronize(); t0 = time.time()
    t_env = time.time(); [TC.step(s, TC.compact_legal(s.legal)[0][:, 0]) for _ in range(5)]; torch.cuda.synchronize()
    t_env = (time.time() - t_env) / 5
    t_net = time.time(); [g.run(TC.encode(s), TC.compact_legal(s.legal)[0]) for _ in range(5)]; torch.cuda.synchronize()
    t_net = (time.time() - t_net) / 5
    t0 = time.time(); G.search(g, s, G.SearchConfig(sims=sims)); torch.cuda.synchronize(); dt = time.time() - t0
    print(f"B={B} sims={sims} net {channels}x{blocks}: one move for all games {dt:.2f}s -> {B / dt:.0f} plies/s, "
          f"{B * sims / dt / 1e3:.0f}k evals/s   (per sim: env step {t_env * 1e3:.0f}ms, network {t_net * 1e3:.0f}ms)")


if __name__ == "__main__":
    if "bench" in sys.argv:
        for B in (1024, 2048):
            bench(B)
        bench(2048, channels=64, blocks=6)
    else:
        test_basic_and_mate()
        test_agrees_with_cpu_search()
