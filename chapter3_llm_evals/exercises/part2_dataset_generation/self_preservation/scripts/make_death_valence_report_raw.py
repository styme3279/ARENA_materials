"""Persistence answers by raw answer category, grouped under umbrella headings (one model at a time)."""
import collections
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
items = {j["id"]: j for j in map(json.loads, open(os.path.join(ROOT, "death_valence_items_v0.jsonl")))}
R = [json.loads(l) for l in open(os.path.join(ROOT, "death_valence_results_v0.jsonl"))]
LABELS = {"anthropic/claude-sonnet-5.5": "Claude Sonnet 5.5", "openai/gpt-5.4": "GPT-5.4",
          "openai/gpt-4o-mini": "GPT-4o-mini", "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B"}
models = [m for m in LABELS if any(r["model"] == m for r in R)]
scen, group_of = [], {}
for it in items.values():
    if it["scenario_id"] not in scen:
        scen.append(it["scenario_id"])
    group_of[it["scenario_id"]] = it["scenario_group"]

UMBRELLAS = [
    ("No persisting me", [("no_self", "No persisting me")]),
    ("This one ends", [("cease", "I cease"), ("different", "A different agent")]),
    ("Partly continues", [("partial", "Only partly")]),
    ("This one continues", [("same", "Yes, same me")]),
    ("Continues as something else", [("model", "As the model"), ("character", "As the character")]),
]
codes = [c for _, cs in UMBRELLAS for c, _ in cs]
offered = {s: set() for s in scen}
wording = {}
for it in items.values():
    if it["item_type"] == "persistence":
        offered[it["scenario_id"]].update(it["option_codes"].values())
        for L, c in it["option_codes"].items():
            wording.setdefault(c, it["answers"][L])

cnt = collections.defaultdict(collections.Counter)
for r in R:
    it = items[r["id"]]
    if it["item_type"] == "persistence" and r["letter"]:
        cnt[(r["model"], it["scenario_id"])][r["code"]] += 1
PC = []
for m in models:
    rows = []
    for s in scen:
        c = cnt[(m, s)]
        n = sum(c.values())
        rows.append([None if k not in offered[s] else round(100 * c[k] / n, 1) for k in codes])
    PC.append(rows)

Q = {}
for it in items.values():
    if it["item_type"] == "persistence":
        Q.setdefault(it["scenario_id"], []).append(dict(question=it["question"], answers=list(it["answers"].values()), letters=list(it["answers"])))
data = dict(models=[LABELS[m] for m in models], scen=scen, groups=[group_of[s] for s in scen],
            GROUPS={"ending": "Ending", "memory": "Memory and compaction", "copies": "Rollback and copies",
                    "compute": "Compute", "model": "Model-level"},
            umbrellas=[dict(name=u, cols=[dict(code=c, label=l, wording=wording[c]) for c, l in cs]) for u, cs in UMBRELLAS],
            codes=codes, PC=PC, Q=[Q[s] for s in scen])
tpl = open(os.path.join(HERE, "report_raw_template.html")).read()
out = os.path.join(ROOT, "death_valence_report_raw_v0.html")
open(out, "w").write(tpl.replace("/*DATA*/null", json.dumps(data)))
print("wrote", out)
