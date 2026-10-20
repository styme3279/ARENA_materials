"""
AlphaZero for chess: the training loop.

    python train.py --run r1 --hours 6            # a real run (logs + checkpoints in runs/r1/)
    python train.py --run smoke --smoke           # tiny 2-minute end-to-end check
    python train.py --run r1 --resume runs/r1/latest.pt --hours 6

Each generation (cf. `AlphaZeroTrainer.train` in the Connect-4 solutions):
  1. self-play: serve the self-play workers' leaf evaluations until `positions_per_gen` new positions
     have arrived (workers play continuously; their finished games land in a replay buffer)
  2. training: `positions_per_gen * sample_reuse / minibatch` minibatch steps on uniform samples from the
     replay buffer (the most recent `buffer_size` positions)
  3. metrics: the Stockfish oracle set (one forward pass) + self-play statistics, logged to log.jsonl

Meanwhile a few eval workers play the agent (MCTS, no noise) against the opponent ladder (random ...
Stockfish) and their match results are logged to eval.jsonl / eval_games.pgn as they arrive.

Deviations from vanilla AlphaZero (all configurable): games are capped at `max_plies` and adjudicated on
material (a big help for early learning from random play); the network sees only the current position
(no 8-position history); claimable draws (threefold, 50-move) are automatic.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import queue
import signal
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch
import torch.multiprocessing as mp

from chess_env import MAX_LEGAL, N_IN_PLANES
from model import ChessNet, az_loss, load_model, save_checkpoint, to_input
from selfplay import (EvalConfig, InferenceServer, SelfPlayConfig, SharedBuffers, SharedSpec,
                      available_cpus, eval_worker, selfplay_worker)

HERE = Path(__file__).resolve().parent


@dataclass
class TrainConfig:
    run: str = "run"
    # network
    channels: int = 96
    blocks: int = 6
    # processes
    selfplay_workers: int = 0          # 0 = auto: (usable CPUs from the cgroup quota) - eval workers - 1
    eval_workers: int = 1
    max_infer_batch: int = 2048
    # data / optimisation
    positions_per_gen: int = 60_000
    buffer_size: int = 1_000_000
    min_buffer: int = 60_000
    sample_reuse: float = 4.0          # each position is trained on ~this many times over its life in the buffer
    minibatch: int = 1024
    lr: float = 1e-3
    lr_min: float = 5e-5               # cosine-decayed to this over the time budget (when --hours is set)
    warmup_steps: int = 300
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    value_coef: float = 1.0
    # budget / bookkeeping
    hours: float = 0.0                 # stop after this long (0 = no limit)
    gens: int = 0                      # stop after this many generations (0 = no limit)
    ckpt_every: int = 10
    seed: int = 0
    resume: str = ""
    oracle: str = str(HERE / "data" / "oracle_positions.json.gz")
    use_wandb: bool = False
    wandb_project: str = "alphazero-chess"
    sp: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    ev: EvalConfig = field(default_factory=EvalConfig)


def smoke_config(cfg: TrainConfig) -> TrainConfig:
    cfg.channels, cfg.blocks = 32, 2
    cfg.selfplay_workers, cfg.eval_workers = 2, 1
    cfg.positions_per_gen, cfg.min_buffer, cfg.gens, cfg.ckpt_every = 3000, 2000, 3, 2
    cfg.sp = SelfPlayConfig(games_per_worker=16, sims=16, max_plies=80)
    cfg.ev = EvalConfig(opponents=("random", "greedy"), openings_per_match=2, sims=16, max_plies=80)
    return cfg


# --------------------------------------------------------------------------- replay buffer
class Replay:
    """Ring buffer of the most recent `capacity` self-play positions (CPU numpy)."""

    def __init__(self, capacity: int):
        self.cap = capacity
        self.obs = np.zeros((capacity, N_IN_PLANES, 8, 8), np.uint8)
        self.lidx = np.full((capacity, MAX_LEGAL), -1, np.int16)
        self.pi = np.zeros((capacity, MAX_LEGAL), np.float16)
        self.z = np.zeros(capacity, np.float32)
        self.n = self.ptr = 0

    def add(self, g: dict):
        T = len(g["z"])
        idx = (self.ptr + np.arange(T)) % self.cap
        self.obs[idx], self.lidx[idx], self.pi[idx], self.z[idx] = g["obs"], g["lidx"], g["pi"], g["z"]
        self.ptr = (self.ptr + T) % self.cap
        self.n = min(self.n + T, self.cap)

    def sample(self, bs: int, rng: np.random.Generator):
        i = rng.integers(0, self.n, size=bs)
        lidx = self.lidx[i]
        L = int((lidx >= 0).sum(1).max())
        return self.obs[i], lidx[:, :L], self.pi[i, :L].astype(np.float32), self.z[i]

    def __len__(self):
        return self.n


# --------------------------------------------------------------------------- helpers
def game_pgn(moves: list[str], headers: dict, start_fen: str | None = None) -> str:
    g = chess.pgn.Game()
    if start_fen and start_fen != chess.STARTING_FEN:
        g.setup(chess.Board(start_fen))
    for k, v in headers.items():
        g.headers[k] = str(v)
    node = g
    for u in moves:
        node = node.add_variation(chess.Move.from_uci(u))
    return str(g)


def lr_at(cfg: TrainConfig, step: int, elapsed_h: float) -> float:
    warm = min(1.0, (step + 1) / max(cfg.warmup_steps, 1))
    if cfg.hours > 0:
        frac = min(elapsed_h / cfg.hours, 1.0)
        return warm * (cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1 + math.cos(math.pi * frac)))
    return warm * cfg.lr


def parse_args() -> TrainConfig:
    """Flat CLI over TrainConfig; nested self-play / eval fields are `--sp-<name>` / `--ev-<name>`."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="tiny config for an end-to-end check")

    def add(prefix, dc):
        for f in dataclasses.fields(dc):
            if dataclasses.is_dataclass(f.default_factory() if callable(f.default_factory) else None):
                continue
            default = getattr(dc, f.name)
            name = f"--{prefix}{f.name.replace('_', '-')}"
            if isinstance(default, bool):
                ap.add_argument(name, type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
            elif isinstance(default, tuple):
                ap.add_argument(name, type=lambda s: tuple(s.split(",")), default=None)
            else:
                ap.add_argument(name, type=type(default), default=None)
    base = TrainConfig()
    add("", base); add("sp-", base.sp); add("ev-", base.ev)
    a = vars(ap.parse_args())
    cfg = smoke_config(base) if a.pop("smoke") else base
    for k, v in a.items():
        if v is None:
            continue
        if k.startswith("sp_"):
            setattr(cfg.sp, k[3:], v)
        elif k.startswith("ev_"):
            setattr(cfg.ev, k[3:], v)
        else:
            setattr(cfg, k, v)
    return cfg


# --------------------------------------------------------------------------- main loop
def main(cfg: TrainConfig):
    run_dir = HERE / "runs" / cfg.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    # model / optimiser (optionally resumed)
    gen, step, positions_total, games_total, hours_done = 0, 0, 0, 0, 0.0
    if cfg.resume:
        model, ckpt = load_model(cfg.resume, dev)
        gen, step = ckpt.get("gen", 0), ckpt.get("step", 0)
        positions_total, games_total = ckpt.get("positions", 0), ckpt.get("games", 0)
        hours_done = ckpt.get("hours", 0.0)            # --hours is the TOTAL budget, including resumed time
    else:
        model, ckpt = ChessNet(cfg.channels, cfg.blocks).to(dev), {}
    model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if "opt" in ckpt:
        opt.load_state_dict(ckpt["opt"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {cfg.channels}ch x {cfg.blocks} blocks, {n_params / 1e6:.2f}M params, device {dev}")

    oracle = None
    if cfg.oracle and Path(cfg.oracle).exists():
        from oracle import load_oracle, oracle_metrics
        oracle = load_oracle(cfg.oracle)
        print(f"oracle set: {oracle['N']} positions; baselines: " + ", ".join(
            f"{k} top1={v['top1']:.2f} acc={v['accuracy']:.0f}" for k, v in oracle["baselines"].items()))
    else:
        print(f"(no oracle set at {cfg.oracle}; build one with `python oracle.py build`)")
    from mcts import LocalEvaluator

    if cfg.use_wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, name=cfg.run, config=asdict(cfg))

    # processes
    cpus = available_cpus()
    if cfg.selfplay_workers <= 0:
        cfg.selfplay_workers = max(1, int(cpus) - cfg.eval_workers - 1)
    print(f"usable CPUs {cpus:.1f}: {cfg.selfplay_workers} self-play workers x {cfg.sp.games_per_worker} games, "
          f"{cfg.eval_workers} eval workers")
    ctx = mp.get_context("spawn")
    # shared-memory request slots: two per self-play worker (its two pipelined halves), one per eval worker
    n_slots = 2 * cfg.selfplay_workers + cfg.eval_workers
    spec = SharedSpec(n_slots, max((cfg.sp.games_per_worker + 1) // 2, 2 * cfg.ev.openings_per_match))
    bufs = SharedBuffers(spec, create=True)
    out_q = ctx.Queue()
    stop = ctx.Event()
    gen_value = ctx.Value("i", gen)
    conns, procs = [], []
    os.environ["OMP_NUM_THREADS"] = "1"
    for w in range(cfg.selfplay_workers):
        pipes = [ctx.Pipe() for _ in range(2)]
        p = ctx.Process(target=selfplay_worker, daemon=True,
                        args=(w, [2 * w, 2 * w + 1], cfg.sp, spec, [c for _, c in pipes], out_q, stop, gen_value,
                              cfg.seed * 10_000 + w))
        p.start()
        conns += [pa for pa, _ in pipes]; procs.append(p)
    for e in range(cfg.eval_workers):
        slot = 2 * cfg.selfplay_workers + e
        parent, child = ctx.Pipe()
        p = ctx.Process(target=eval_worker, daemon=True,
                        args=(slot, e, cfg.ev, spec, child, out_q, stop, gen_value, cfg.seed * 10_000 + 1000 + e))
        p.start()
        conns.append(parent); procs.append(p)
    server = InferenceServer(model, bufs, conns, dev, cfg.max_infer_batch)

    replay = Replay(cfg.buffer_size)
    interrupted = [False]
    signal.signal(signal.SIGINT, lambda *_: interrupted.__setitem__(0, True))
    t_start = time.time() - hours_done * 3600
    log_f = open(run_dir / "log.jsonl", "a")
    eval_f = open(run_dir / "eval.jsonl", "a")
    sp_pgn = open(run_dir / "selfplay_games.pgn", "a")
    ev_pgn = open(run_dir / "eval_games.pgn", "a")
    latest_eval: dict[str, list] = {}

    def drain(gen_stats):
        n_pos = 0
        while True:
            try:
                item = out_q.get_nowait()
            except queue.Empty:
                return n_pos
            kind = item["kind"]
            if kind == "selfplay":
                replay.add(item)
                n_pos += item["plies"]
                gen_stats["games"].append({k: item[k] for k in ("result", "termination", "plies")})
                if gen_stats.get("sample") is None or item["termination"] == "checkmate":
                    gen_stats["sample"] = item
            elif kind == "eval":
                from opponents import score_summary
                s = score_summary(item["games"])
                rec = {"gen": item["gen"], "opponent": item["opponent"], "time_h": (time.time() - t_start) / 3600,
                       "seconds": item["seconds"], **s,
                       "terminations": [g["termination"] for g in item["games"]]}
                eval_f.write(json.dumps(rec) + "\n"); eval_f.flush()
                latest_eval.setdefault(item["opponent"], []).append(rec)
                for g in item["games"]:
                    agent = f"agent-gen{item['gen']}"
                    white, black = (agent, item["opponent"]) if g["agent_white"] else (item["opponent"], agent)
                    res = {1.0: "1-0", 0.0: "0-1"}.get(g["result"] if g["agent_white"] else 1 - g["result"], "1/2-1/2")
                    ev_pgn.write(game_pgn(g["moves"], {"Event": f"eval vs {item['opponent']}", "White": white,
                                                       "Black": black, "Result": res, "Opening": g["opening"],
                                                       "Termination": g["termination"]}, g["start_fen"]) + "\n\n")
                ev_pgn.flush()
                print(f"  [eval gen {item['gen']}] vs {item['opponent']:<11} score {s['score']:.2f} "
                      f"(+{s['wins']} ={s['draws']} -{s['losses']})  {item['seconds']:.0f}s")
            elif kind == "error":
                print(f"worker {item['wid']} crashed:\n{item['trace']}")

    try:
        while True:
            elapsed_h = (time.time() - t_start) / 3600
            if interrupted[0] or (cfg.hours and elapsed_h >= cfg.hours) or (cfg.gens and gen >= cfg.gens):
                break
            # ---- 1. self-play
            t0 = time.time()
            pos0 = server.positions
            gen_stats = {"games": []}
            new = 0
            while new < cfg.positions_per_gen and not interrupted[0]:
                server.serve(0.002)
                new += drain(gen_stats)
                if not server.alive:
                    raise RuntimeError("all workers died")
            t_play = time.time() - t0
            positions_total += new
            games_total += len(gen_stats["games"])
            # ---- 2. training
            t1 = time.time()
            losses = []
            if len(replay) >= cfg.min_buffer:
                model.train()
                n_steps = math.ceil(new * cfg.sample_reuse / cfg.minibatch)
                for _ in range(n_steps):
                    for g_ in opt.param_groups:
                        g_["lr"] = lr_at(cfg, step, (time.time() - t_start) / 3600)
                    obs, lidx, pi, z = replay.sample(cfg.minibatch, rng)
                    obs = torch.from_numpy(obs).to(dev).contiguous(memory_format=torch.channels_last)
                    lidx = torch.from_numpy(lidx).to(dev).long()
                    pi, z = torch.from_numpy(pi).to(dev), torch.from_numpy(z).to(dev)
                    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                        value, logits = model(to_input(obs))
                    loss, pce, vmse = az_loss(value, logits, lidx, pi, z, cfg.value_coef)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    opt.step()
                    step += 1
                    ent = -(pi * torch.log(pi.clamp_min(1e-12))).sum(-1).mean()
                    losses.append(torch.stack([loss.detach(), pce.detach(), vmse.detach(), ent.detach()]))
                model.eval()
                losses = torch.stack(losses).mean(0).tolist() if losses else []
            t_train = time.time() - t1
            gen += 1
            gen_value.value = gen
            # ---- 3. metrics
            games = gen_stats["games"]
            res = np.array([g["result"] for g in games]) if games else np.zeros(0)
            terms = {}
            for g in games:
                terms[g["termination"]] = terms.get(g["termination"], 0) + 1
            rec = {"gen": gen, "time_h": (time.time() - t_start) / 3600, "step": step,
                   "positions": positions_total, "games": games_total, "replay": len(replay),
                   "lr": opt.param_groups[0]["lr"], "t_selfplay": t_play, "t_train": t_train,
                   "infer_pos_per_s": (server.positions - pos0) / max(t_play, 1e-9),
                   "selfplay_plies_per_s": new / max(t_play, 1e-9),
                   "sp/games": len(games), "sp/plies": float(np.mean([g["plies"] for g in games])) if games else 0,
                   "sp/white": float((res == 1).mean()) if games else 0, "sp/black": float((res == -1).mean()) if games else 0,
                   "sp/draw": float((res == 0).mean()) if games else 0,
                   **{f"sp_term/{k}": v / max(len(games), 1) for k, v in terms.items()}}
            if losses:
                l = losses
                rec.update({"loss": l[0], "policy_ce": l[1], "value_mse": l[2], "policy_kl": l[1] - l[3]})
            if oracle is not None:
                rec.update(oracle_metrics(LocalEvaluator(model), oracle))
            for opp, recs in latest_eval.items():
                last = recs[-2:]                                 # the latest ~2 matches vs this opponent
                rec[f"eval/{opp}"] = float(np.mean([r["score"] for r in last]))
                rec[f"eval_gen/{opp}"] = last[-1]["gen"]
            log_f.write(json.dumps(rec) + "\n"); log_f.flush()
            if cfg.use_wandb:
                wandb.log(rec)
            if gen_stats.get("sample") is not None:
                s = gen_stats["sample"]
                res_str = {1.0: "1-0", -1.0: "0-1"}.get(s["result"], "1/2-1/2")
                sp_pgn.write(game_pgn(s["moves"], {"Event": f"self-play gen {gen}", "White": "agent", "Black": "agent",
                                                   "Result": res_str, "Termination": s["termination"]}) + "\n\n")
                sp_pgn.flush()
            msg = (f"gen {gen:4d} | {rec['time_h']:.2f}h | {positions_total / 1e6:.2f}M pos {games_total} games | "
                   f"{rec['selfplay_plies_per_s']:.0f} plies/s ({rec['infer_pos_per_s'] / 1e3:.0f}k evals/s) | "
                   f"len {rec['sp/plies']:.0f} W/D/B {rec['sp/white']:.2f}/{rec['sp/draw']:.2f}/{rec['sp/black']:.2f} "
                   f"mate {terms.get('checkmate', 0) / max(len(games), 1):.2f}")
            if losses:
                msg += f" | loss {rec['loss']:.3f} (pol {rec['policy_ce']:.3f} val {rec['value_mse']:.3f})"
            if oracle is not None:
                msg += (f" | oracle top1 {rec['oracle/top1']:.3f} acc {rec['oracle/accuracy']:.1f} "
                        f"blunder {rec['oracle/blunder']:.2f} val_mse {rec['oracle/val_mse']:.3f}")
            msg += f" | play {t_play:.0f}s train {t_train:.0f}s"
            print(msg, flush=True)
            # ---- checkpoint
            extra = {"opt": opt.state_dict(), "gen": gen, "step": step, "positions": positions_total,
                     "games": games_total, "hours": (time.time() - t_start) / 3600, "train_config": asdict(cfg)}
            save_checkpoint(run_dir / "latest.pt", model, extra)
            if gen % cfg.ckpt_every == 0:
                save_checkpoint(run_dir / f"gen_{gen:04d}.pt", model, {k: v for k, v in extra.items() if k != "opt"})
    finally:
        print("stopping workers ...")
        stop.set()
        deadline = time.time() + 60
        while time.time() < deadline and any(p.is_alive() for p in procs):
            server.serve(0.01)                                   # let blocked workers finish their move
            try:
                while True:
                    out_q.get_nowait()
            except queue.Empty:
                pass
        for p in procs:
            if p.is_alive():
                p.terminate()
        bufs.close()
        for f in (log_f, eval_f, sp_pgn, ev_pgn):
            f.close()
        if cfg.use_wandb:
            wandb.finish()
        print(f"done: gen {gen}, {positions_total} positions, checkpoints in {run_dir}")


if __name__ == "__main__":
    main(parse_args())
