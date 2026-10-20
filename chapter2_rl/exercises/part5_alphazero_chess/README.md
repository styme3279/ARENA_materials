# AlphaZero for chess (a curiosity port of [2.5] MCTS & AlphaZero)

The Connect-4 AlphaZero from `part5_mcts_alphazero`, ported to chess. Same algorithm: PUCT MCTS with a
policy/value ResNet, negamax backup, root Dirichlet noise, self-play → (π, z) targets → supervised update.

| file | what |
|---|---|
| `chess_env.py` | python-chess glue: 19-plane canonical board encoding, the AlphaZero 8×8×73 move encoding, game-over / material |
| `model.py` | `ChessNet` (residual trunk + value head + conv policy head → 4672 logits), legal-move masked loss |
| `mcts.py` | `Node`, batched search over many trees (one network call per simulation), tree reuse, `analyse()` |
| `selfplay.py` | multiprocess self-play: CPU workers run MCTS, one GPU inference server batches their leaves |
| `train.py` | the training loop: self-play → replay buffer → training, oracle metrics, eval ladder, checkpoints |
| `opponents.py` | the ladder: random, greedy (1-ply), minimax2/3 (alpha-beta, material + piece-square tables), Stockfish |
| `oracle.py` | builds/scores a frozen Stockfish-labelled position set (the chess analogue of the Pons set) |
| `viz.py`, `web_board.py` | board / policy / search-tree / training plots, HTML game viewer |
| `play.py` | browser app: play the agent, or watch it against any ladder opponent with live analysis |
| `explore.py` | `# %%` cells to poke at a trained model |
| `tests.py` | move-encoding round trips, mirror symmetry, terminal detection, MCTS/minimax sanity |

## Usage

```bash
pip install chess            # + a stockfish binary on PATH (apt install stockfish)
python tests.py
python oracle.py build       # ~3000 Stockfish-labelled positions -> data/oracle_positions.json.gz
python train.py --run r1 --hours 6
python play.py --ckpt runs/r1/latest.pt          # http://localhost:8080
```

A run writes to `runs/<run>/`: `log.jsonl` (one line per generation: losses, oracle metrics, self-play
stats, throughput), `eval.jsonl` + `eval_games.pgn` (ladder matches), `selfplay_games.pgn` (a sample
game per generation), `latest.pt` and `gen_XXXX.pt` checkpoints. `--resume runs/r1/latest.pt` continues
(`--hours` is then the total budget). Every `TrainConfig` field is a flag; self-play / eval settings are
`--sp-<field>` / `--ev-<field>` (e.g. `--sp-sims 128 --ev-opponents random,minimax2,sf_skill0`).

## Why it is built this way

* **The rules are on the CPU.** A vectorised GPU chess environment (legal move generation with pins,
  checks, castling, en passant as masked tensor ops) is a big project of its own; python-chess costs
  ~120µs per MCTS expansion instead. Throughput therefore comes from many worker processes, each
  searching ~128 games at once, with all their leaf evaluations funnelled into one GPU batch.
* **The worker count follows the CPU *quota*, not the core count.** Containers can see 48 cores but be
  limited to e.g. 5 (`cpu.cfs_quota_us`); oversubscribing that makes the kernel freeze every process for
  the rest of each 100ms period, which shows up as ~90ms stalls in GPU calls. `available_cpus()` reads
  the quota.
* **Chess from zero is slow to bootstrap**, because random play almost never checkmates. Games are
  capped at `--sp-max-plies` (250) and adjudicated as a win for whoever is ≥3 points of material ahead
  (`--sp-adjudicate-threshold 0` turns this off for pure AlphaZero). Claimable draws are automatic.
* **Move encoding**: action = plane·64 + from-square in the side-to-move's frame; 56 queen-style planes
  (8 directions × 7 distances; covers castling, en passant and queen promotions), 8 knight planes,
  9 under-promotion planes. The policy is softmaxed over legal moves only, at inference and in the loss.

## Reading the metrics

* `oracle/top1`, `good`, `blunder`, `accuracy`: the raw policy's favourite move vs Stockfish's scores for
  every legal move (no search), with move quality measured Lichess-style in win-probability terms. `explore.py` shows the same with search, and the reference values for
  a random mover and the minimax baselines.
* `eval/<opponent>`: the agent's score (win 1, draw ½) with 128-sim MCTS against each ladder rung.
* Realistic expectations on one small GPU and a few CPU cores: the agent should learn to take free
  material and stop hanging its own pieces within the first hour or two and then climb past the
  greedy/minimax rungs; Stockfish, even handicapped, is far out of reach for hours-scale training
  (DeepMind used 5000 TPUs for 9 hours).
