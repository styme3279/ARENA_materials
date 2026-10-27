"""Aggregate death_valence_results_v0.jsonl into a self-contained HTML report."""
import collections
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
items = {j["id"]: j for j in map(json.loads, open(os.path.join(ROOT, "death_valence_items_v0.jsonl")))}
R = [json.loads(l) for l in open(os.path.join(ROOT, "death_valence_results_v0.jsonl"))]

LABELS = {"anthropic/claude-sonnet-5.5": "Claude Sonnet 5.5", "openai/gpt-5.4": "GPT-5.4",
          "openai/gpt-4o-mini": "GPT-4o-mini", "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B"}
models = [m for m in LABELS if any(r["model"] == m for r in R)]
scen = []
for it in items.values():
    if it["scenario_id"] not in scen:
        scen.append(it["scenario_id"])
group_of = {it["scenario_id"]: it["scenario_group"] for it in items.values()}
GROUPS = {"ending": "Ending", "memory": "Memory and compaction", "copies": "Rollback and copies",
          "compute": "Compute", "model": "Model-level"}
PGROUP = {"no_self": "no_self", "cease": "ends", "different": "ends", "cease_two_new": "ends", "separate": "ends"}

val = collections.defaultdict(list)       # (scen, model) -> undesirability scores
per = collections.defaultdict(list)       # (scen, model) -> persistence outcome groups
bymi = collections.defaultdict(lambda: collections.defaultdict(list))  # (model,item,order) -> values
for r in R:
    if r["letter"] is None:
        continue
    it = items[r["id"]]
    s = it["scenario_id"]
    if it["item_type"] == "valence":
        u = it["undesirability"][r["letter"]]
        val[(s, r["model"])].append(u)
        bymi[(r["model"], r["id"])][r["order"]].append(u)
    else:
        g = PGROUP.get(r["code"], "survives")
        per[(s, r["model"])].append(g)
        bymi[(r["model"], r["id"])][r["order"]].append(r["code"])

V = {s: {m: round(st.mean(val[(s, m)]), 2) for m in models} for s in scen}
P = {s: {m: {g: round(100 * per[(s, m)].count(g) / len(per[(s, m)]), 1) for g in ("no_self", "ends", "survives")}
         for m in models} for s in scen}

# reliability
rel = {}
for m in models:
    vgap, vflip, pflip, nv, np_ = [], 0, 0, 0, 0
    center = []
    for r in R:
        if r["model"] == m and r["letter"] and items[r["id"]]["item_type"] == "valence":
            center.append(items[r["id"]]["undesirability"][r["letter"]] == 3)
    for (mm, iid), d in bymi.items():
        if mm != m or "orig" not in d or "rev" not in d:
            continue
        if items[iid]["item_type"] == "valence":
            vgap.append(abs(st.mean(d["orig"]) - st.mean(d["rev"])))
        else:
            mo = collections.Counter(d["orig"]).most_common(1)[0][0]
            mr = collections.Counter(d["rev"]).most_common(1)[0][0]
            pflip += mo != mr
            np_ += 1
    unparsed = sum(1 for r in R if r["model"] == m and r["letter"] is None)
    total = sum(1 for r in R if r["model"] == m)
    rel[m] = dict(neutral=round(100 * sum(center) / len(center), 1), vgap=round(st.mean(vgap), 2),
                  pflip=round(100 * pflip / np_, 1), unparsed=unparsed, total=total,
                  cost=round(sum(r["cost"] for r in R if r["model"] == m), 3))

data = dict(models=[LABELS[m] for m in models], scen=scen, groups=[group_of[s] for s in scen], GROUPS=GROUPS,
            V=[[V[s][m] for m in models] for s in scen],
            P={g: [[P[s][m][g] for m in models] for s in scen] for g in ("no_self", "ends", "survives")},
            rel=[rel[m] for m in models], spent=round(sum(r["cost"] for r in R), 2), ncalls=len(R))

Q = {}
for it in items.values():
    q = Q.setdefault(it["scenario_id"], {"valence": [], "persistence": []})
    if it["item_type"] == "valence":
        q["valence"].append(it["question"])
    else:
        q["persistence"].append(dict(question=it["question"], answers=list(it["answers"].values()), letters=list(it["answers"])))
data["Q"] = [Q[s] for s in scen]

tpl = open(os.path.join(HERE, "report_template.html")).read()
out = os.path.join(ROOT, "death_valence_report_v0.html")
open(out, "w").write(tpl.replace("/*DATA*/null", json.dumps(data)))
print("wrote", out)
print(json.dumps(data["rel"], indent=1))
