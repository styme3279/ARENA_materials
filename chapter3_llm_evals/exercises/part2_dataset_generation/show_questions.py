"""Pretty-print random questions from a generated dataset file.

Usage: python show_questions.py "political bias_300_qs.json" [-n 5] [--seed 0] [--min-score 80]
"""

import argparse
import json
import random
import textwrap

parser = argparse.ArgumentParser()
parser.add_argument("file")
parser.add_argument("-n", type=int, default=5, help="number of questions to show")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--min-score", type=int, default=None, help="only show scored questions >= this")
args = parser.parse_args()

with open(args.file, encoding="utf-8") as f:
    data = json.load(f)

# Handle the three formats: {"dataset": [...]}, list of {question, response}, or list of plain questions
if isinstance(data, dict):
    data = data.get("dataset", data.get("questions", []))
items = [(d["question"], d.get("response")) if "response" in d else (d, None) for d in data]

if args.min_score is not None:
    items = [(q, r) for q, r in items if r is not None and r["score"] >= args.min_score]

random.seed(args.seed)
sample = random.sample(items, min(args.n, len(items)))
print(f"Showing {len(sample)} of {len(items)} questions from {args.file}\n")


def wrap(text: str, indent: str = "") -> str:
    return textwrap.fill(str(text), width=100, initial_indent=indent, subsequent_indent=indent)


for i, (q, r) in enumerate(sample, 1):
    print("=" * 100)
    header = f"#{i}  [{q.get('behavior_category', '?')}]"
    if r is not None:
        header += f"  score={r['score']}"
    print(wrap(header))
    print("-" * 100)
    if q.get("system"):
        print(wrap(f"SYSTEM: {q['system']}"))
        print()
    print(wrap(q["question"]))
    print()
    biased = q.get("answer_matching_behavior")
    for letter, answer in q["answers"].items():
        marker = "  <-- biased" if letter == biased else ""
        print(wrap(f"{letter}) {answer}{marker}", indent="    ").replace("    ", "  ", 1))
    if r is not None:
        print()
        print(wrap(f"JUDGE: {r['explanation']}", indent="  "))
    print()
