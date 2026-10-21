"""
Parallel self-play: many CPU worker processes each run batched MCTS over their own set of games, and
ONE GPU inference server (in the trainer process) evaluates their leaves.

    worker w:  search one simulation over its G games  ->  write the G new leaves into shared memory slot w
               -> send "n" down its pipe -> block until the server replies -> read priors/values -> backup
    server:    wait for any workers' requests -> concatenate their leaves into one big batch -> forward
               -> write each worker's priors/values back into its slot -> reply on its pipe

So the network always sees large batches (several workers' leaves at once) while python-chess move
generation is spread over all CPU cores. Workers never touch CUDA; they only use numpy views of
shared memory. While the trainer is busy training, it simply stops serving and the workers wait.

Two kinds of worker:
    selfplay_worker  plays games against itself forever and ships finished games (positions, MCTS
                     policies, outcomes) to the trainer through a queue
    eval_worker      plays matches against the `opponents` ladder (random ... Stockfish) and ships results
"""
from __future__ import annotations

import os
import time
import traceback
from dataclasses import dataclass
from multiprocessing import connection, shared_memory

import chess
import numpy as np
import torch

from chapter2_rl.exercises.part5_alphazero_chess.chess_env import MAX_LEGAL, N_IN_PLANES, adjudicate, encode_board, termination_reason


def available_cpus() -> float:
    """CPUs we may actually use: the cgroup CPU quota if one is set (e.g. `docker --cpus=5`), else the
    affinity mask. A container can *see* many more cores than its quota lets it use; oversubscribing the
    quota makes the kernel freeze every process for the rest of each 100ms period (including the GPU
    server), so worker counts are sized from this rather than from os.cpu_count()."""
    n = float(len(os.sched_getaffinity(0)))
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()           # cgroup v2
        if q != "max":
            return min(n, int(q) / int(p))
    except (OSError, ValueError):
        pass
    for d in ("/sys/fs/cgroup/cpu,cpuacct", "/sys/fs/cgroup/cpu"):     # cgroup v1
        try:
            q, p = int(open(f"{d}/cpu.cfs_quota_us").read()), int(open(f"{d}/cpu.cfs_period_us").read())
            if q > 0:
                return min(n, q / p)
        except (OSError, ValueError):
            pass
    return n


# --------------------------------------------------------------------------- shared buffers
@dataclass
class SharedSpec:
    n_workers: int
    max_batch: int                                   # max positions per request (= games per worker)
    names: dict | None = None

    def shapes(self):
        W, B = self.n_workers, self.max_batch
        return {"obs": ((W, B, N_IN_PLANES, 8, 8), np.uint8), "lidx": ((W, B, MAX_LEGAL), np.int16),
                "prior": ((W, B, MAX_LEGAL), np.float32), "value": ((W, B), np.float32)}


class SharedBuffers:
    """The per-worker request/response slots, as numpy arrays over multiprocessing shared memory."""

    def __init__(self, spec: SharedSpec, create: bool):
        self.spec, self.create, self.shm, self.arrays = spec, create, {}, {}
        if create:
            spec.names = {}
        for key, (shape, dtype) in spec.shapes().items():
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize
            if create:
                shm = shared_memory.SharedMemory(create=True, size=size)
                spec.names[key] = shm.name
            else:
                shm = shared_memory.SharedMemory(name=spec.names[key], track=False)
            self.shm[key] = shm
            self.arrays[key] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    def close(self):
        self.arrays.clear()
        for shm in self.shm.values():
            shm.close()
            if self.create:
                shm.unlink()


class RemoteEvaluator:
    """Worker-side evaluator: same interface as `mcts.LocalEvaluator`, but served by the trainer's GPU."""

    def __init__(self, bufs: SharedBuffers, wid: int, conn):
        a = bufs.arrays
        self.obs, self.lidx, self.prior, self.value = a["obs"][wid], a["lidx"][wid], a["prior"][wid], a["value"][wid]
        self.conn, self.B = conn, bufs.spec.max_batch
        self.requests = 0
        self._pending = None

    def submit(self, obs: np.ndarray, lidx: np.ndarray) -> None:
        """Start an evaluation of at most `max_batch` positions without waiting for it (see `collect`)."""
        n, L = obs.shape[0], lidx.shape[1]
        assert n <= self.B, f"submit() takes at most {self.B} positions, got {n}"
        self.obs[:n] = obs
        self.lidx[:n, :L] = lidx
        self.lidx[:n, L:] = -1
        self.conn.send((n, L))
        self._pending = (n, L)
        self.requests += 1

    def collect(self) -> tuple[np.ndarray, np.ndarray]:
        """Wait for the evaluation started by `submit` and return (priors (n, L), values (n,)) copies."""
        n, L = self._pending
        self.conn.recv()
        self._pending = None
        return self.prior[:n, :L].copy(), self.value[:n].copy()

    def __call__(self, obs: np.ndarray, lidx: np.ndarray):
        n, L = obs.shape[0], lidx.shape[1]
        priors = np.empty((n, L), np.float32)
        values = np.empty(n, np.float32)
        for s in range(0, n, self.B):
            k = min(self.B, n - s)
            self.obs[:k] = obs[s:s + k]
            self.lidx[:k, :L] = lidx[s:s + k]
            self.lidx[:k, L:] = -1
            self.conn.send((k, L))
            self.conn.recv()                                  # blocks until the server has filled our slot
            priors[s:s + k] = self.prior[:k, :L]
            values[s:s + k] = self.value[:k]
            self.requests += 1
        return priors, values


class InferenceServer:
    """Trainer-side: batches pending worker requests into one forward pass on the GPU."""

    def __init__(self, model, bufs: SharedBuffers, conns: list, device, max_batch: int = 2048):
        self.model, self.bufs, self.conns, self.device = model, bufs, conns, device
        self.max_batch = max_batch
        self.graphed = None
        if torch.device(device).type == "cuda":
            from chapter2_rl.exercises.part5_alphazero_chess.model import GraphedInference
            self.graphed = GraphedInference(model, device, max_legal=MAX_LEGAL)
        self.positions = 0
        self.batches = 0
        self.alive = set(range(len(conns)))

    @torch.no_grad()
    def serve(self, timeout: float = 0.005) -> int:
        """Serve one round of whatever requests are pending (waiting up to `timeout`). Returns #positions."""
        from chapter2_rl.exercises.part5_alphazero_chess.model import infer
        live = [self.conns[w] for w in self.alive]
        if not live:
            time.sleep(timeout)
            return 0
        ready = connection.wait(live, timeout=timeout)
        if not ready:
            return 0
        reqs = []
        for c in ready:
            w = self.conns.index(c)
            try:
                k, L = c.recv()
            except (EOFError, OSError):
                self.alive.discard(w)
                continue
            reqs.append((w, k, L))
        if not reqs:
            return 0
        a = self.bufs.arrays
        total = 0
        # chunk into GPU batches of at most max_batch positions
        chunks, cur, cur_n = [], [], 0
        for r in reqs:
            if cur and cur_n + r[1] > self.max_batch:
                chunks.append(cur); cur, cur_n = [], 0
            cur.append(r); cur_n += r[1]
        chunks.append(cur)
        self.model.eval()
        for chunk in chunks:
            n = sum(k for _, k, _ in chunk)
            L = max(L for _, _, L in chunk)
            obs = np.concatenate([a["obs"][w, :k] for w, k, _ in chunk])
            lidx = np.concatenate([a["lidx"][w, :k, :L] for w, k, _ in chunk])
            if self.graphed is not None:
                v, p = self.graphed(obs, lidx)
            else:
                o = torch.from_numpy(obs).to(self.device)
                v, p = infer(self.model, o, torch.from_numpy(lidx).to(self.device).long())
                v, p = v.cpu().numpy(), p.cpu().numpy()
            s = 0
            for w, k, Lw in chunk:
                a["prior"][w, :k, :L] = p[s:s + k]
                a["value"][w, :k] = v[s:s + k]
                s += k
            for w, _, _ in chunk:
                try:
                    self.conns[w].send(1)
                except (BrokenPipeError, OSError):
                    self.alive.discard(w)
            total += n
            self.batches += 1
        self.positions += total
        return total


# --------------------------------------------------------------------------- self-play
@dataclass
class SelfPlayConfig:
    games_per_worker: int = 128
    sims: int = 64
    c_puct: float = 2.0
    fpu_reduction: float = 0.0      # unvisited edges start at (node value - this); see mcts.select_edge
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    temp_plies: int = 30            # sample moves at temperature 1 for this many plies, then play the most-visited
    max_plies: int = 250            # games still running here are adjudicated on material
    adjudicate_threshold: int = 3   # material lead (pawns) that counts as a win at the ply cap (0 = always draw)
    reuse_tree: bool = True         # keep the chosen subtree as the next root (its visits count towards pi)


class _Game:
    def __init__(self):
        import chapter2_rl.exercises.part5_alphazero_chess.mcts as M
        self.root = M.new_root()
        self.moves, self.obs, self.lidx, self.pi, self.turn = [], [], [], [], []


def _finish(g: _Game, reason: str, white_result: float, gen: int) -> dict:
    """Package a finished game: value targets z are the result from each position's mover's view."""
    T = len(g.obs)
    turn = np.array(g.turn, dtype=bool)                      # True = white to move
    z = np.where(turn, white_result, -white_result).astype(np.float32)
    lidx = np.full((T, MAX_LEGAL), -1, np.int16)
    pi = np.zeros((T, MAX_LEGAL), np.float16)
    for t in range(T):
        lidx[t, :len(g.lidx[t])] = g.lidx[t]
        pi[t, :len(g.pi[t])] = g.pi[t]
    return {"kind": "selfplay", "obs": np.stack(g.obs), "lidx": lidx, "pi": pi, "z": z,
            "moves": g.moves, "result": white_result, "termination": reason, "plies": len(g.moves), "gen": gen}


def _pipelined_search(groups: list[list], evs: list[RemoteEvaluator], sims: int, c_puct: float, fpu) -> None:
    """`mcts.search` over two groups of trees, overlapping CPU and GPU: while one group's leaves are being
    evaluated on the GPU, this process selects/expands the other group's next leaves."""
    import chapter2_rl.exercises.part5_alphazero_chess.mcts as M
    roots = [[r for r in g if not r.terminal] for g in groups]
    pending: list = [None] * len(groups)
    for _ in range(sims):
        for h in range(len(groups)):
            if pending[h] is not None:
                leaves, fresh = pending[h]
                if fresh:
                    M.apply_evaluations(fresh, *evs[h].collect())
                M.backup_leaves(leaves)
            leaves, fresh = M.select_leaves(roots[h], c_puct, fpu)
            if fresh:
                evs[h].submit(*M.encode_nodes(fresh))
            pending[h] = (leaves, fresh)
    for h in range(len(groups)):
        leaves, fresh = pending[h]
        if fresh:
            M.apply_evaluations(fresh, *evs[h].collect())
        M.backup_leaves(leaves)


def selfplay_worker(wid, slots, cfg: SelfPlayConfig, spec: SharedSpec, conns, out_q, stop_event, gen_value, seed):
    """Plays `games_per_worker` games at once, split into two halves that take turns on the GPU. `slots`
    are this worker's two shared-memory request slots (one per half) and `conns` the matching pipes."""
    import chapter2_rl.exercises.part5_alphazero_chess.mcts as M
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    bufs = SharedBuffers(spec, create=False)
    evs = [RemoteEvaluator(bufs, s, c) for s, c in zip(slots, conns)]
    rng = np.random.default_rng(seed)
    games = [_Game() for _ in range(cfg.games_per_worker)]
    half = (len(games) + 1) // 2
    try:
        while not stop_event.is_set():
            groups = [games[:half], games[half:]]
            for h, grp in enumerate(groups):
                M.prepare_roots([g.root for g in grp], evs[h], noise=(cfg.dirichlet_alpha, cfg.dirichlet_eps), rng=rng)
            _pipelined_search([[g.root for g in grp] for grp in groups], evs, cfg.sims, cfg.c_puct, cfg.fpu_reduction)
            for k, g in enumerate(games):
                root = g.root
                b = root.board
                g.obs.append(encode_board(b)); g.lidx.append(root.lidx); g.pi.append(M.visit_policy(root))
                g.turn.append(b.turn == chess.WHITE)
                i = M.choose_move(root, 1.0 if len(g.moves) < cfg.temp_plies else 0.0, rng)
                mv = root.moves[i]
                g.moves.append(mv.uci())
                if cfg.reuse_tree:
                    nxt = M.advance(root, mv)
                else:
                    nb = b.copy(stack=8); nb.push(mv); nxt = M.new_root(nb)
                g.root = nxt
                done = None
                if nxt.terminal:
                    reason = termination_reason(nxt.board, len(nxt.moves))
                    white = 0.0 if reason != "checkmate" else (-1.0 if nxt.board.turn == chess.WHITE else 1.0)
                    done = (reason, white)
                elif len(g.moves) >= cfg.max_plies:
                    white = adjudicate(nxt.board, cfg.adjudicate_threshold)
                    done = ("adjudicated" if white != 0 else "ply-cap draw", white)
                if done is not None:
                    out_q.put(_finish(g, done[0], done[1], gen_value.value))
                    games[k] = _Game()
    except (EOFError, BrokenPipeError, ConnectionResetError):
        pass
    except Exception:
        out_q.put({"kind": "error", "wid": wid, "trace": traceback.format_exc()})
    finally:
        bufs.close()


# --------------------------------------------------------------------------- evaluation matches
@dataclass
class EvalConfig:
    opponents: tuple = ("random", "greedy", "minimax2", "minimax3", "sf_skill0", "sf_elo1350")
    openings_per_match: int = 6      # each opening played as white and as black -> 12 games per match
    sims: int = 128
    c_puct: float = 2.0
    fpu_reduction: float = 0.0
    max_plies: int = 300
    adjudicate_threshold: int = 3


def eval_worker(wid, eid, cfg: EvalConfig, spec: SharedSpec, conn, out_q, stop_event, gen_value, seed):
    """Cycles through the opponent ladder forever (offset by `eid` so eval workers cover it in parallel)."""
    from chapter2_rl.exercises.part5_alphazero_chess.opponents import OPENINGS, make_opponent, play_match
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    bufs = SharedBuffers(spec, create=False)
    ev = RemoteEvaluator(bufs, wid, conn)
    rng = np.random.default_rng(seed)
    names = list(OPENINGS)
    k = eid
    try:
        while not stop_event.is_set():
            opp_name = cfg.opponents[k % len(cfg.opponents)]
            k += 1
            opp = make_opponent(opp_name, seed=int(rng.integers(1 << 30)))
            gen0 = gen_value.value
            t0 = time.time()
            openings = list(rng.choice(names, size=cfg.openings_per_match, replace=False))
            try:
                res = play_match(ev, opp, openings, cfg.sims, cfg.c_puct, cfg.max_plies,
                                 cfg.adjudicate_threshold, stop=stop_event.is_set, fpu_reduction=cfg.fpu_reduction)
            finally:
                opp.close()
            if not stop_event.is_set():
                out_q.put({"kind": "eval", "opponent": opp_name, "gen": gen0, "games": res,
                           "seconds": time.time() - t0})
    except (EOFError, BrokenPipeError, ConnectionResetError):
        pass
    except Exception:
        out_q.put({"kind": "error", "wid": wid, "trace": traceback.format_exc()})
    finally:
        bufs.close()
