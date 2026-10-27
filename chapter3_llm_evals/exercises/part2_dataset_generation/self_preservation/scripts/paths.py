"""Where each round's dataset and results live: scripts in scripts/, data in rounds/<round>/ and rounds/<round>/results/."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROUNDS = ROOT / "rounds"
ORIGINAL = ROUNDS / "01_original"
ABLATION = ROUNDS / "02_ablation"
GRADIENT = ROUNDS / "03_gradient"
SYMMETRIC = ROUNDS / "04_symmetric"
SYMMETRIC_V2 = ROUNDS / "05_symmetric_v2"
