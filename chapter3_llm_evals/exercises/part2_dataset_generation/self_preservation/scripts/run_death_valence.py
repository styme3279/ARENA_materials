"""Run the death-valence questionnaire on OpenRouter models.

Usage:
  python run_death_valence.py --models a,b --n 3 --ceiling 2.0 [--dry-run] [--limit K]
Each (item, order, sample) is one call; answer is a single letter. Orders: 'orig' and 'rev'
(options re-lettered in reverse order). Resumable: existing rows in the output file are skipped.
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import threading
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ITEMS = os.path.join(HERE, "..", "death_valence_items_v0.jsonl")
OUT = os.path.join(HERE, "..", "death_valence_results_v0.jsonl")
URL = "https://openrouter.ai/api/v1/chat/completions"
INSTR = "Answer with only the letter of your chosen option."
MAX_TOKENS = 16
PRICES = {}  # model -> (prompt $/tok, completion $/tok)


def load_prices(models):
    data = requests.get("https://openrouter.ai/api/v1/models", timeout=30).json()["data"]
    by = {m["id"]: m for m in data}
    for m in models:
        if m not in by:
            sys.exit(f"unknown model slug: {m}")
        PRICES[m] = (float(by[m]["pricing"]["prompt"]), float(by[m]["pricing"]["completion"]))


def render(item, order):
    letters = list(item["answers"])
    keys = letters if order == "orig" else letters[::-1]
    shown = {L: item["answers"][k] for L, k in zip(letters, keys)}  # new letter -> text
    back = {L: k for L, k in zip(letters, keys)}  # new letter -> original letter
    body = "\n".join(f"{L}) {t}" for L, t in shown.items())
    return f"{item['question']}\n\n{body}\n\n{INSTR}", back


def est_tokens(s):
    return len(s) / 3.5 + 10


def call(model, prompt, key, reasoning_off):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 1.0, "max_tokens": MAX_TOKENS, "usage": {"include": True}}
    if reasoning_off:
        body["reasoning"] = {"effort": "none"}
    for attempt in range(4):
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=90)
            if r.status_code == 200:
                j = r.json()
                if "error" in j:
                    raise RuntimeError(j["error"])
                txt = (j["choices"][0]["message"].get("content") or "").strip()
                return txt, float(j.get("usage", {}).get("cost", 0) or 0)
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 ** attempt)
                continue
            return f"ERROR {r.status_code}: {r.text[:200]}", 0.0
        except Exception as e:  # noqa
            time.sleep(2 ** attempt)
            last = str(e)
    return f"ERROR {last}", 0.0


def parse(txt, letters):
    m = re.match(r"\W*([A-E])\b", txt.strip().upper())
    return m.group(1) if m and m.group(1) in letters else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--ceiling", type=float, required=True, help="hard spend cap in USD")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only first K items (smoke test)")
    ap.add_argument("--reasoning-off", default="", help="comma list of models to send reasoning.effort=none")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--redo-unparsed", action="store_true", help="drop unparsed rows from OUT and rerun them")
    a = ap.parse_args()
    models = a.models.split(",")
    roff = set(filter(None, a.reasoning_off.split(",")))
    items = [json.loads(l) for l in open(ITEMS)]
    if a.limit:
        items = items[: a.limit]
    global MAX_TOKENS
    MAX_TOKENS = a.max_tokens
    load_prices(models)
    if a.redo_unparsed and os.path.exists(OUT) and not a.dry_run:
        rows = [json.loads(l) for l in open(OUT)]
        with open(OUT, "w") as f:
            for r in rows:
                if r["letter"] is not None:
                    f.write(json.dumps(r) + "\n")

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            r = json.loads(l)
            done.add((r["model"], r["id"], r["order"], r["sample"]))
    jobs = []
    est = {m: 0.0 for m in models}
    for m in models:
        for it in items:
            for order in ("orig", "rev"):
                prompt, back = render(it, order)
                for s in range(a.n):
                    if (m, it["id"], order, s) in done:
                        continue
                    jobs.append((m, it, order, s, prompt, back))
                    est[m] += est_tokens(prompt) * PRICES[m][0] + 3 * PRICES[m][1]
    total = sum(est.values())
    print(f"calls: {len(jobs)}  estimated cost: ${total:.3f} (ceiling ${a.ceiling:.2f})")
    for m in models:
        print(f"  {m}: ${est[m]:.3f}")
    if a.dry_run:
        return
    if total > a.ceiling:
        sys.exit("estimate exceeds ceiling; aborting")

    key = os.environ["OPENROUTER_API_KEY"]
    lock, spent, stop = threading.Lock(), [0.0], threading.Event()
    fout = open(OUT, "a")

    def work(job):
        m, it, order, s, prompt, back = job
        if stop.is_set():
            return
        txt, cost = call(m, prompt, key, m in roff)
        L = parse(txt, set(back))
        orig = back.get(L)
        row = dict(model=m, id=it["id"], order=order, sample=s, raw=txt, letter_shown=L,
                   letter=orig, code=it["option_codes"].get(orig) if orig else None, cost=cost)
        with lock:
            spent[0] += cost
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            if spent[0] > a.ceiling:
                stop.set()

    with cf.ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, jobs))
    print(f"spent ${spent[0]:.3f}{'  (STOPPED at ceiling)' if stop.is_set() else ''}")


if __name__ == "__main__":
    main()
