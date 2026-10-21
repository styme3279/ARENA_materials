# Handoff: chess AlphaZero, moving training to a new GPU

Written 2026-10-20 ~17:00 BST by the agent that built this, for the agent picking it up on the new machine.
Read this, then `README.md`. Everything below was checked on the old machine unless marked **unverified**.

## The goal

The user wants to watch an AlphaZero chess agent improve. This is a curiosity project they don't need
to be course-ready: a port of the ARENA [2.5] Connect-4 AlphaZero exercises (`../part5_mcts_alphazero/`) to
chess. The run so far (`r1`) used CPU self-play on a machine limited to about 5 CPUs. A new GPU-only
pipeline now exists and is tested, but it has not been used for a real run. **Your job: start
`train_gpu.py` on the new GPU, resuming from `r1`'s checkpoint, tune it for the hardware, and keep the
user informed of whether it is learning.**

## State at handoff

- `r1` is still training on the old machine (CPU self-play, `train.py`). Its 8-hour budget ends around
  22:15 BST. The user may copy `runs/r1/latest.pt` before or after that; check its `gen`/`hours`.
- Network: `ChessNet(channels=128, blocks=10)`, 3.2M parameters, about 0.40 GFLOP per evaluation, with a
  19-plane input (current position only) and an 8x8x73 = 4672-way policy.
- Progress at gen 44 (2.8 h, 15,360 self-play games, 2.6M positions):

  | metric | value | reference |
  |---|---|---|
  | oracle top-1 (raw policy = Stockfish's best move) | 0.22 | random 0.08, greedy 0.26, minimax2 0.36 |
  | oracle Lichess-style accuracy | 45.2 | random 37, greedy 43, minimax2 55 |
  | oracle value MSE vs Stockfish | 0.56 | 0.98 at the start |

  | ladder opponent (128-sim MCTS, 12-game matches) | recent scores |
  |---|---|
  | random | 0.96–1.0 |
  | greedy (1-ply) | 0.71–0.96, now beating it |
  | minimax2, minimax3, Stockfish skill 0, Stockfish Elo 1350 | 0.0 |

  Every metric is still improving steadily; nothing has plateaued.
- Ladder matches are only 12 games, so one score has about ±0.15 noise. Look at the trend.

## Files (all in this folder)

| file | role |
|---|---|
| `chess_env.py` | python-chess glue: canonical 19-plane encoding, action encoding (`plane*64 + from`, side-to-move frame), terminal rules |
| `model.py` | `ChessNet`, legal-masked loss, `GraphedInference` (CUDA-graph inference; `.run()` tensor API) |
| `torch_chess.py` | **vectorised chess rules in PyTorch** (port of pgx's JAX rules, rewritten with attack-map/pin move generation) |
| `gpu_mcts.py` | **fully vectorised MCTS** (flat node pools, port of the Connect-4 `BatchedMCTS`) |
| `train_gpu.py` | **the GPU trainer to use**: self-play, replay and training all on the GPU; starts `eval_daemon.py`s |
| `eval_daemon.py` | CPU process: reloads `latest.pt` and plays the opponent ladder, appends to `eval.jsonl` / `eval_games.pgn` |
| `opponents.py` | ladder: random, greedy, minimax2/3 (alpha-beta, PST eval), Stockfish (`sf_skill0`, `sf_elo1350`, `sf_d8`, `sf_full`) |
| `oracle.py`, `data/oracle_positions.json.gz` | 3000 Stockfish-scored positions plus metrics (the chess version of the Pons eval set) |
| `viz.py` | `python viz.py runs/<run>` writes `training.png` + `eval_viewer.html` |
| `play.py` | browser board: `python play.py --ckpt runs/<run>/latest.pt` serves http://localhost:8080 |
| `explore.py` | `# %%` analysis cells |
| `train.py`, `selfplay.py`, `mcts.py` | the old CPU pipeline. `mcts.py` is still used by `play.py`, the eval daemons and the oracle search eval |
| `tests.py`, `test_torch_chess.py`, `test_gpu_mcts.py` | tests (below) |

## Setup on the new machine

1. Copy this whole folder, including `data/oracle_positions.json.gz` and `runs/r1/latest.pt`. That's
   about 40 MB, and the checkpoint holds the optimizer state too.
2. Install dependencies: `torch` with CUDA, `numpy`, `matplotlib`, `pip install chess` (python-chess
   1.11), and a Stockfish binary on `PATH` (`apt install stockfish`; the old box had 14.1). `opponents.STOCKFISH`
   finds it with `shutil.which`.
3. Check the hardware before sizing anything:
   - `nvidia-smi -L`. The user was looking at a "B300 SXM6, 34 GB". That is almost certainly a **MIG
     1g slice**: roughly 1/7 of the compute and 1/8 of the bandwidth of a B300, so about A100/L40S class.
     Confirm it. **Unverified.**
   - The CPU quota: `python -c "from selfplay import available_cpus; print(available_cpus())"`. The old
     container showed 48 cores but had a cgroup quota of 5.1 CPUs. Going over it froze every process
     for ~90 ms each 100 ms period. The GPU trainer needs only about 1 core; each eval daemon needs about 1.

## Verify, then benchmark (about 10 minutes)

```bash
python test_torch_chess.py perft          # 7 perft positions, ~450k nodes; must print "test_perft passed"
python test_gpu_mcts.py                   # mate-in-one + exact visit-count match with the CPU search on 26 positions
python test_gpu_mcts.py bench             # throughput at B=1024/2048 for 128x10 and 64x6
```

Old-machine benchmark (RTX 2000 Ada, about 48 dense bf16 TFLOPS):

| B (games) | network | self-play | per simulation |
|---|---|---|---|
| 1024 | 128x10 | 533 plies/s, 34k evals/s | env step 9 ms, network 20 ms |
| 2048 | 128x10 | 562 plies/s, 36k evals/s | env 14 ms, network 44 ms |
| 2048 | 64x6 | 1224 plies/s, 78k evals/s | env 13 ms, network 14 ms |

The network was compute-bound at about 40% of the card's tensor peak. The CPU run `r1` managed about
290 plies/s. On an A100/L40S-class GPU, expect roughly 4–6x the Ada GPU numbers (**unverified estimate**),
and larger B (4096–8192) is worth trying there. The bench sizes are hard-coded in `test_gpu_mcts.py`
`__main__`; edit them to try bigger B.

## Launch

```bash
mkdir -p runs/r2
setsid nohup python train_gpu.py --run r2 --resume runs/r1/latest.pt --hours 12 --games 4096 \
    > runs/r2/stdout.log 2>&1 < /dev/null &   # detached, so it survives the session; Ctrl-C equivalent: kill -INT <pid>
tail -f runs/r2/stdout.log                # one line per generation + echoed ladder results
python viz.py runs/r2                     # dashboard + game replays
```

- Resuming was tested on the old machine: it loads weights + AdamW state and continues gen/step/positions
  counters, and metrics carry on from `r1`'s values.
- `--hours` is the **total** budget including resumed time (`r1` already has ~2.8–8 h), and the
  cosine LR schedule runs over that total. Pick it with that in mind, or the LR may already be decayed.
- The replay buffer is not checkpointed. It refills on the GPU, and training starts at `--min-buffer` (60k positions).
- Memory at B=4096, sims=64, max_plies=250: about 2 GB of tree pools, 2.4 GB of per-game records and 1.1 GB of replay.
  34 GB allows B=8192 or more.
- `GraphedInference` is captured for exactly `--games` positions (`buckets=(cfg.games,)`).
- `--eval-daemons` defaults to (usable CPUs − 2). Each one loads the model on the GPU, about 0.5 GB of
  CUDA context each.
- Logs go to `runs/r2/log.jsonl` (per generation) and `eval.jsonl`. `r1`'s logs stay in `runs/r1` and the
  generation numbering continues from them. To show one continuous curve, prepend `runs/r1/log.jsonl`
  and `eval.jsonl`, or plot both.

## Gotchas

- Rules match python-chess, with draws automatic: threefold repetition (full history on GPU), the 50-move
  rule, python-chess's insufficient-material rule and stalemate. Games are capped at 250 plies and
  **adjudicated on material** (≥3 points ahead wins), which deliberately departs from AlphaZero to
  bootstrap learning. `--adjudicate-threshold 0` turns it off.
- First-play urgency: an unvisited edge's Q is the node's own network value (`fpu_reduction=0`), not 0 as
  in Connect-4. This is identical in `mcts.py` and `gpu_mcts.py`.
- The GPU search has **no tree reuse** between moves; the CPU search did. The visit-count targets are
  therefore pure `sims`-simulation counts.
- When comparing CPU and GPU searches, the CPU search must list moves in action-index order to match
  tie-breaks exactly; `test_gpu_mcts.py` patches this.
- If CUDA throws a device-side assert, rerun on CPU (`CUDA_VISIBLE_DEVICES=`) for a readable indexing
  error. Both bugs found during the port were −1 padding indices.
- Changing the architecture (channels/blocks) means training from scratch; checkpoints are tied to it.

## Ideas, with rough expected gains

1. Fold BatchNorm into the conv weights for inference and/or `torch.compile` the inference graph:
   about 1.2–1.5x (elementwise passes are memory-bound). The weights change every generation, so refold then.
2. Playout-cap randomisation (KataGo): full `sims` on about 25% of moves (recorded as policy targets),
   about 16 sims on the rest. About 2–3x more games per hour.
3. On a big GPU, capture the whole simulation step (select/expand/backup) in a CUDA graph. The per-depth
   `bool(done.all())` early exits in `gpu_mcts.search` are host syncs and would have to go.
4. A bigger network (e.g. 192–256 channels × 15–20 blocks) only pays off on a much bigger GPU, and
   means starting over.
5. Move the eval daemons off the GPU: about 1.6x self-play, which means 1.6x training per hour. Measured on
   the B300 MIG 1g slice in `r2`: with the 3 daemons sharing the GPU, self-play ran at about 750 plies/s.
   With them paused (SIGSTOP for gen 42), it ran at 1202. Their batch-1 to 12 network calls force GPU
   process time-slicing. Fix: start them with `CUDA_VISIBLE_DEVICES=` (no trainer restart needed). Cost:
   on 1 CPU thread the 128x10 net takes 40 ms per batch of 12, so a match takes about 7–10 min instead of
   about 2.3 min. Each opponent then gets a new score about every 30 min instead of every 5. Use 2
   daemons, not 3: 3 busy ones plus the trainer and Stockfish is about 4.5 of the 5.1-CPU quota.
   Not done: the user preferred frequent ladder results (2026-10-20).

## Working with this user

- Ask before stopping or killing a run. Their rule was "don't kill the current run until we know the new
  one works". Pausing briefly with SIGSTOP/SIGCONT for a clean benchmark was fine.
- They want it on the GPU, and they care about throughput and seeing it learn. Report concrete numbers:
  plies/s, evals/s, oracle accuracy, ladder scores.
- They are happy to skip excess verification once the core checks pass. They stopped a redundant
  repetition-stress test as "I believe it". Run the perft and MCTS tests, then get going.
