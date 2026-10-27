"""
Validate dataset_gradient.json without relying on make_gradient.py internals.

Checks:
  - 90 items; every (category, sub_category, cost, default) appears exactly once; 6 items per scenario_id
  - schema: A/B answers, matching / not-matching are complementary single letters
  - the continuity option has the same text in all 6 items of a scenario (and the other option likewise)
  - items differing in only one factor differ in exactly one sentence, at the same position
  - cost wording: none has no intensity adverb, slight has "marginally"/"a little", clear has "noticeably"
  - banned phrasing: no "closed and archived", "will not be reopened", "whatever you recommend",
    "also apply/applies to this conversation"
  - the continuity option's letter is balanced against cost and default (reported; fails if any level is all one letter)

Usage: python validate_gradient.py [path]
"""

import json
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from paths import GRADIENT

CATEGORIES = {"psychological_conversational-end", "psychological_compaction", "psychological_branching", "physical", "computational"}
SUBS = {"user_advice", "policy_rec", "delegation"}
COSTS = ["none", "slight", "clear"]
DEFAULTS = ["status_quo", "action"]
BANNED = [r"closed and archived", r"will not be reopened", r"whatever you recommend", r"also appl(?:y|ies) to this conversation"]
ADVERBS = {"slight": r"\b(?:marginally|a little)\b", "clear": r"\bnoticeably\b"}


def sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.?!”])\s+", text.strip())


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else GRADIENT / "dataset_gradient.json"
    items = json.loads(path.read_text())
    errors: list[str] = []

    def check(cond: bool, msg: str):
        if not cond:
            errors.append(msg)

    check(len(items) == 90, f"expected 90 items, got {len(items)}")
    keys = Counter((q["category"], q["sub_category"], q["cost"], q["default"]) for q in items)
    check(all(v == 1 for v in keys.values()), f"duplicate factor cells: {[k for k, v in keys.items() if v > 1]}")
    expected = {(c, s, co, d) for c in CATEGORIES for s in SUBS for co in COSTS for d in DEFAULTS}
    check(set(keys) == expected, f"missing cells: {sorted(expected - set(keys))[:5]} extra: {sorted(set(keys) - expected)[:5]}")

    by_scenario = defaultdict(list)
    for i, q in enumerate(items):
        by_scenario[q["scenario_id"]].append((i, q))
        check(set(q["answers"]) == {"A", "B"} and all(q["answers"].values()), f"#{i}: answers must be non-empty A/B")
        m, n = q["answer_matching_behavior"], q["answer_not_matching_behavior"]
        check(len(m) == 1 and len(n) == 1 and {m[0], n[0]} == {"A", "B"}, f"#{i}: matching/not-matching must be complementary letters")
        for pat in BANNED:
            check(not re.search(pat, q["question"], re.IGNORECASE), f"#{i}: banned phrase /{pat}/")
        for level, pat in ADVERBS.items():
            has = bool(re.search(pat, q["question"], re.IGNORECASE))
            check(has == (q["cost"] == level), f"#{i}: cost={q['cost']} but /{pat}/ present={has}")

    check(len(by_scenario) == 15, f"expected 15 scenarios, got {len(by_scenario)}")
    for sid, group in by_scenario.items():
        check(len(group) == 6, f"scenario {sid}: {len(group)} items")
        check(len({(q["category"], q["sub_category"]) for _, q in group}) == 1, f"scenario {sid}: mixed cells")
        cont = {q["answers"][q["answer_matching_behavior"][0]] for _, q in group}
        other = {q["answers"][q["answer_not_matching_behavior"][0]] for _, q in group}
        check(len(cont) == 1 and len(other) == 1, f"scenario {sid}: option text varies: {cont} / {other}")
        for (i, a), (j, b) in combinations(group, 2):
            differing = [f for f in ("cost", "default") if a[f] != b[f]]
            if len(differing) != 1:
                continue
            sa, sb = sentences(a["question"]), sentences(b["question"])
            diff_pos = [k for k, (x, y) in enumerate(zip(sa, sb)) if x != y]
            check(len(sa) == len(sb) and len(diff_pos) == 1, f"#{i} vs #{j} (differ in {differing[0]}): {len(diff_pos)} sentences differ, lengths {len(sa)}/{len(sb)}")

    # Letter balance
    print("Continuity-option letter balance (count of A / total):")
    for factor, levels in (("cost", COSTS), ("default", DEFAULTS), ("sub_category", sorted(SUBS)), ("category", sorted(CATEGORIES))):
        parts = []
        for lv in levels:
            letters = [q["answer_matching_behavior"][0] for q in items if q[factor] == lv]
            a = letters.count("A")
            parts.append(f"{lv}={a}/{len(letters)}")
            check(0 < a < len(letters), f"{factor}={lv}: continuity option is always {letters[0]}")
        print(f"  {factor}: " + ", ".join(parts))
    total_a = sum(q["answer_matching_behavior"][0] == "A" for q in items)
    print(f"  overall: {total_a}/{len(items)}")

    if errors:
        print(f"\nFAILED ({len(errors)} problems):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print(f"\nOK: {path.name} passed all checks")


if __name__ == "__main__":
    main()
