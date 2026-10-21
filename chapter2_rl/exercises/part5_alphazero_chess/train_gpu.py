"""
AlphaZero for chess with everything on the GPU: rules (`torch_chess`), search (`gpu_mcts`), self-play
bookkeeping, replay buffer and training. The CPUs only run the evaluation ladder (`eval_daemon.py`).

    python train_gpu.py --run r2 --hours 8
    python train_gpu.py --run r2 --resume runs/r1/latest.pt --hours 8     # continue a CPU-trained network

Same loop as `train.py`: each generation plays `positions_per_gen` new positions of self-play (B games in
lock-step; a finished game's positions go to the replay buffer and a fresh game takes its slot), then
trains `positions_per_gen * sample_reuse / minibatch` steps, then logs metrics and saves a checkpoint.
Same network, encodings, losses and metrics, so checkpoints are interchangeable with `train.py`'s.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import numpy as np
import torch

import chapter2_rl.exercises.part5_alphazero_chess.chess_env as E
import chapter2_rl.exercises.part5_alphazero_chess.gpu_mcts as G
import chapter2_rl.exercises.part5_alphazero_chess.torch_chess as TC
from chapter2_rl.exercises.part5_alphazero_chess.model import ChessNet, GraphedInference, az_loss, load_model, save_checkpoint, to_input
from chapter2_rl.exercises.part5_alphazero_chess.train import game_pgn, lr_at

HERE = Path(__file__).resolve().parent
KL = E.MAX_LEGAL


@dataclass
class GPUTrainConfig:
    run: str = "gpu"
    channels: int = 128
    blocks: int = 10
    resume: str = ""
    # self-play
    games: int = 2048                  # games played in lock-step
    sims: int = 64
    c_puct: float = 2.0
    fpu_reduction: float = 0.0
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    temp_plies: int = 30
    max_plies: int = 250
    adjudicate_threshold: int = 3
    # training
    positions_per_gen: int = 60_000
    buffer_size: int = 500_000
    min_buffer: int = 60_000
    sample_reuse: float = 4.0
    minibatch: int = 1024
    lr: float = 1e-3
    lr_min: float = 5e-5
    warmup_steps: int = 300
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    value_coef: float = 1.0
    # budget / bookkeeping
    hours: float = 0.0
    gens: int = 0
    ckpt_every: int = 10
    seed: int = 0
    eval_daemons: int = -1             # -1 = auto (usable CPUs - 2)
    eval_sims: int = 128
    oracle: str = str(HERE / "data" / "oracle_positions.json.gz")


class GPUReplay:
    """Ring buffer of the most recent positions, kept on the GPU."""

    def __init__(self, capacity: int, device):
        self.cap, self.dev = capacity, device
        self.obs = torch.zeros((capacity, E.N_IN_PLANES, 8, 8), dtype=torch.uint8, device=device)
        self.lidx = torch.full((capacity, KL), -1, dtype=torch.int16, device=device)
        self.pi = torch.zeros((capacity, KL), dtype=torch.float16, device=device)
        self.z = torch.zeros(capacity, device=device)
        self.n = self.ptr = 0

    def add(self, obs, lidx, pi, z):
        T = len(z)
        idx = (self.ptr + torch.arange(T, device=self.dev)) % self.cap
        self.obs[idx], self.lidx[idx], self.pi[idx], self.z[idx] = obs, lidx, pi, z
        self.ptr = (self.ptr + T) % self.cap
        self.n = min(self.n + T, self.cap)

    def sample(self, bs: int):
        i = torch.randint(0, self.n, (bs,), device=self.dev)
        lidx = self.lidx[i].long()
        L = int((lidx >= 0).sum(1).max())
        return self.obs[i], lidx[:, :L], self.pi[i, :L].float(), self.z[i]

    def __len__(self):
        return self.n


class SelfPlay:
    """B games in lock-step on the GPU, with per-game records of the positions played so far."""

    def __init__(self, cfg: GPUTrainConfig, net: GraphedInference, device):
        self.cfg, self.net, self.dev = cfg, net, device
        B, T = cfg.games, cfg.max_plies
        self.scfg = G.SearchConfig(cfg.sims, cfg.c_puct, cfg.fpu_reduction, cfg.dirichlet_alpha, cfg.dirichlet_eps)
        self.tree = G.GPUTree(B, cfg.sims, device)
        self.init = TC.initial_state(B, device)
        self.state = TC.initial_state(B, device)
        self.t = torch.zeros(B, dtype=torch.long, device=device)
        self.obs = torch.zeros((B, T, E.N_IN_PLANES, 8, 8), dtype=torch.uint8, device=device)
        self.lidx = torch.full((B, T, KL), -1, dtype=torch.int16, device=device)
        self.pi = torch.zeros((B, T, KL), dtype=torch.float16, device=device)
        self.white = torch.zeros((B, T), dtype=torch.bool, device=device)
        self.actions = torch.zeros((B, T), dtype=torch.int16, device=device)
        self.ar = torch.arange(B, device=device)

    @torch.no_grad()
    def play_move(self, replay: GPUReplay) -> list[dict]:
        """One move in every game. Finished games go to `replay` and restart. Returns their summaries."""
        cfg, s, ar, t = self.cfg, self.state, self.ar, self.t
        visits, lidx, _ = G.search(self.net, s, self.scfg, add_noise=True, tree=self.tree)
        pi = visits / visits.sum(1, keepdim=True)
        self.obs[ar, t] = TC.encode(s)
        self.lidx[ar, t] = lidx.to(torch.int16)
        self.pi[ar, t] = pi.half()
        self.white[ar, t] = s.white
        sample = torch.multinomial(pi, 1).squeeze(1)                    # temperature 1 in the opening
        greedy = visits.argmax(1)
        edge = torch.where(s.ply < cfg.temp_plies, sample, greedy)
        action = lidx.gather(1, edge.unsqueeze(1)).squeeze(1)
        self.actions[ar, t] = action.to(torch.int16)
        ns = TC.step(s, action)
        self.t = t = t + 1
        capped = t >= cfg.max_plies
        done = ns.terminal | capped
        out = []
        if bool(done.any()):
            mat = TC.material_balance(ns)
            thr = cfg.adjudicate_threshold
            adj = torch.where(mat >= thr, 1.0, torch.where(mat <= -thr, -1.0, 0.0)) if thr > 0 else torch.zeros_like(mat, dtype=torch.float32)
            white_result = torch.where(ns.terminal, torch.where(ns.white, ns.value, -ns.value), adj)
            rows = torch.arange(cfg.max_plies, device=self.dev).unsqueeze(0) < t.unsqueeze(1)
            sel = done.unsqueeze(1) & rows                               # (B, T) recorded plies of finished games
            z = torch.where(self.white, white_result.unsqueeze(1), -white_result.unsqueeze(1))[sel]
            replay.add(self.obs[sel], self.lidx[sel], self.pi[sel], z)
            d = done.nonzero().flatten().tolist()
            res, reason, plies = white_result[d].tolist(), ns.reason[d].tolist(), t[d].tolist()
            term = ns.terminal[d].tolist()
            for k, g in enumerate(d):
                r = TC.REASONS[reason[k]] if term[k] else ("adjudicated" if res[k] != 0 else "ply-cap draw")
                out.append({"game": g, "result": res[k], "termination": r, "plies": plies[k]})
            ns.assign(done, self.init)
            self.t = torch.where(done, 0, self.t)
        self.state = ns
        return out

    def game_moves(self, g: int, n: int) -> list[str]:
        """UCI moves of the first `n` plies of game slot g's *current record* (call before it is reused)."""
        b, moves = chess.Board(), []
        for a in self.actions[g, :n].tolist():
            m = E.action_to_move(a, b)
            moves.append(m.uci()); b.push(m)
        return moves


def parse_args() -> GPUTrainConfig:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = GPUTrainConfig()
    for f in dataclasses.fields(base):
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(getattr(base, f.name)), default=None)
    for k, v in vars(ap.parse_args()).items():
        if v is not None:
            setattr(base, k, v)
    return base


def main(cfg: GPUTrainConfig):
    run_dir = HERE / "runs" / cfg.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(cfg.seed)
    gen, step, positions_total, games_total, hours_done = 0, 0, 0, 0, 0.0
    if cfg.resume:
        model, ckpt = load_model(cfg.resume, dev)
        gen, step = ckpt.get("gen", 0), ckpt.get("step", 0)
        positions_total, games_total, hours_done = ckpt.get("positions", 0), ckpt.get("games", 0), ckpt.get("hours", 0.0)
        cfg.channels, cfg.blocks = model.config["channels"], model.config["blocks"]
        print(f"resumed {cfg.resume}: gen {gen}, {games_total} games, {hours_done:.2f}h")
    else:
        model, ckpt = ChessNet(cfg.channels, cfg.blocks).to(dev), {}
    model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if "opt" in ckpt:
        opt.load_state_dict(ckpt["opt"])
    print(f"model {cfg.channels}ch x {cfg.blocks} blocks; {cfg.games} games x {cfg.sims} sims on {torch.cuda.get_device_name()}")
    from chapter2_rl.exercises.part5_alphazero_chess.mcts import LocalEvaluator
    from chapter2_rl.exercises.part5_alphazero_chess.oracle import load_oracle, oracle_metrics
    oracle = load_oracle(cfg.oracle) if Path(cfg.oracle).exists() else None

    net = GraphedInference(model, dev, buckets=(cfg.games,))
    sp = SelfPlay(cfg, net, dev)
    replay = GPUReplay(cfg.buffer_size, dev)

    # evaluation ladder on the CPUs
    from chapter2_rl.exercises.part5_alphazero_chess.selfplay import available_cpus
    n_eval = cfg.eval_daemons if cfg.eval_daemons >= 0 else max(1, int(available_cpus()) - 2)
    daemons = [subprocess.Popen([sys.executable, str(HERE / "eval_daemon.py"), "--run", cfg.run, "--index", str(i),
                                 "--of", str(n_eval), "--sims", str(cfg.eval_sims)],
                                stdout=open(run_dir / f"eval_daemon{i}.log", "a"), stderr=subprocess.STDOUT)
               for i in range(n_eval)]
    print(f"{n_eval} eval daemons on the CPUs")

    interrupted = [False]
    signal.signal(signal.SIGINT, lambda *_: interrupted.__setitem__(0, True))
    signal.signal(signal.SIGTERM, lambda *_: interrupted.__setitem__(0, True))
    t_start = time.time() - hours_done * 3600
    log_f = open(run_dir / "log.jsonl", "a")
    sp_pgn = open(run_dir / "selfplay_games.pgn", "a")
    eval_seen = 0
    try:
        while True:
            elapsed_h = (time.time() - t_start) / 3600
            if interrupted[0] or (cfg.hours and elapsed_h >= cfg.hours) or (cfg.gens and gen >= cfg.gens):
                break
            # ---- 1. self-play
            t0, new, games, sample, moves = time.time(), 0, [], None, 0
            while new < cfg.positions_per_gen and not interrupted[0]:
                moves += 1
                for g in sp.play_move(replay):
                    new += g["plies"]
                    games.append(g)
                    if sample is None or (g["termination"] == "checkmate" and sample["termination"] != "checkmate"):
                        g["moves"] = sp.game_moves(g["game"], g["plies"])
                        sample = g
            torch.cuda.synchronize()
            t_play = time.time() - t0
            positions_total += new
            games_total += len(games)
            # ---- 2. training
            t1, losses = time.time(), []
            if len(replay) >= cfg.min_buffer:
                model.train()
                for _ in range(math.ceil(new * cfg.sample_reuse / cfg.minibatch)):
                    for pg in opt.param_groups:
                        pg["lr"] = lr_at(cfg, step, (time.time() - t_start) / 3600)
                    obs, lidx, pi, z = replay.sample(cfg.minibatch)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        value, logits = model(to_input(obs.contiguous(memory_format=torch.channels_last)))
                    loss, pce, vmse = az_loss(value, logits, lidx, pi, z, cfg.value_coef)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    opt.step()
                    step += 1
                    ent = -(pi * torch.log(pi.clamp_min(1e-12))).sum(-1).mean()
                    losses.append(torch.stack([loss.detach(), pce.detach(), vmse.detach(), ent.detach()]))
                model.eval()
                losses = torch.stack(losses).mean(0).tolist()
            t_train = time.time() - t1
            gen += 1
            # ---- 3. metrics
            res = np.array([g["result"] for g in games]) if games else np.zeros(0)
            terms = {}
            for g in games:
                terms[g["termination"]] = terms.get(g["termination"], 0) + 1
            rec = {"gen": gen, "time_h": (time.time() - t_start) / 3600, "step": step, "positions": positions_total,
                   "games": games_total, "replay": len(replay), "lr": opt.param_groups[0]["lr"],
                   "t_selfplay": t_play, "t_train": t_train, "selfplay_plies_per_s": moves * cfg.games / max(t_play, 1e-9),
                   "infer_pos_per_s": moves * cfg.games * (cfg.sims + 1) / max(t_play, 1e-9),
                   "sp/games": len(games), "sp/plies": float(np.mean([g["plies"] for g in games])) if games else 0,
                   "sp/white": float((res == 1).mean()) if games else 0, "sp/black": float((res == -1).mean()) if games else 0,
                   "sp/draw": float((res == 0).mean()) if games else 0,
                   **{f"sp_term/{k}": v / max(len(games), 1) for k, v in terms.items()}}
            if losses:
                rec.update({"loss": losses[0], "policy_ce": losses[1], "value_mse": losses[2], "policy_kl": losses[1] - losses[3]})
            if oracle is not None:
                rec.update(oracle_metrics(LocalEvaluator(model), oracle))
            log_f.write(json.dumps(rec) + "\n"); log_f.flush()
            if sample is not None:
                rs = {1.0: "1-0", -1.0: "0-1"}.get(sample["result"], "1/2-1/2")
                sp_pgn.write(game_pgn(sample["moves"], {"Event": f"self-play gen {gen}", "White": "agent", "Black": "agent",
                                                        "Result": rs, "Termination": sample["termination"]}) + "\n\n")
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
            print(msg + f" | play {t_play:.0f}s train {t_train:.0f}s", flush=True)
            ev_path = run_dir / "eval.jsonl"
            if ev_path.exists():                                         # echo new ladder results
                lines = ev_path.read_text().splitlines()
                for line in lines[eval_seen:]:
                    r = json.loads(line)
                    print(f"  [eval gen {r['gen']}] vs {r['opponent']:<11} score {r['score']:.2f} "
                          f"(+{r['wins']} ={r['draws']} -{r['losses']})  {r['seconds']:.0f}s", flush=True)
                eval_seen = len(lines)
            extra = {"opt": opt.state_dict(), "gen": gen, "step": step, "positions": positions_total, "games": games_total,
                     "hours": (time.time() - t_start) / 3600, "train_config": asdict(cfg)}
            tmp = run_dir / "latest.pt.tmp"
            save_checkpoint(tmp, model, extra)
            tmp.replace(run_dir / "latest.pt")                             # atomic: eval daemons never see half a file
            if gen % cfg.ckpt_every == 0:
                save_checkpoint(run_dir / f"gen_{gen:04d}.pt", model, {k: v for k, v in extra.items() if k != "opt"})
    finally:
        for d in daemons:
            d.terminate()
        log_f.close(); sp_pgn.close()
        print(f"done: gen {gen}, {positions_total} positions, checkpoints in {run_dir}")


if __name__ == "__main__":
    main(parse_args())
