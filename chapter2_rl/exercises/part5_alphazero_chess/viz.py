"""
Visualisation for chess AlphaZero (the chess versions of the Connect-4 `utils` plots):

    board_svg / show_board      python-chess SVG board (renders inline in notebooks), with arrows
    draw_board(ax, board, ...)  the same board drawn with matplotlib, for composing figures
    plot_board_and_policy       position + MCTS visit share vs network prior for the top moves
    plot_mcts_tree              the search tree: edges labelled with SAN, N and Q, width ~ N
    plot_training(run_dir)      loss, oracle metrics vs baselines, ladder scores, self-play stats
    game_viewer_html(games)     a self-contained HTML replay viewer (board, move list, value graph)
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import chess
import chess.svg
import numpy as np

LIGHT, DARK = "#f0d9b5", "#b58863"
GLYPH = {chess.KING: "♚", chess.QUEEN: "♛", chess.ROOK: "♜", chess.BISHOP: "♝",
         chess.KNIGHT: "♞", chess.PAWN: "♟"}


# --------------------------------------------------------------------------- boards
def board_svg(board: chess.Board, moves_weights=(), lastmove=None, size: int = 400, flipped=False) -> str:
    """SVG of `board`; `moves_weights` = [(chess.Move, weight in 0..1)] are drawn as green arrows."""
    arrows = [chess.svg.Arrow(m.from_square, m.to_square, color=f"#15781b{int(40 + 180 * w):02x}")
              for m, w in moves_weights]
    if lastmove is None and board.move_stack:
        lastmove = board.peek()
    check = board.king(board.turn) if board.is_check() else None
    return chess.svg.board(board, arrows=arrows, lastmove=lastmove, size=size, flipped=flipped, check=check)


def show_board(board: chess.Board, moves_weights=(), lastmove=None, size: int = 400, flipped=False):
    """Inline SVG for notebooks (returns an IPython SVG object; outside notebooks, returns the string)."""
    s = board_svg(board, moves_weights, lastmove, size, flipped)
    try:
        from IPython.display import SVG
        return SVG(s)
    except ImportError:
        return s


def draw_board(ax, board: chess.Board, moves_weights=(), lastmove=None, flipped=False, title=None):
    """Draw `board` on a matplotlib axis, with optional move arrows [(chess.Move, weight 0..1)]."""
    import matplotlib.patheffects as pe
    from matplotlib.patches import FancyArrow, Rectangle

    def xy(sq):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        return (7 - f, 7 - r) if flipped else (f, r)

    if lastmove is None and board.move_stack:
        lastmove = board.peek()
    hl = {lastmove.from_square, lastmove.to_square} if lastmove else set()
    for sq in chess.SQUARES:
        x, y = xy(sq)
        light = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1
        color = ("#cdd26a" if light else "#aaa23a") if sq in hl else (LIGHT if light else DARK)
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color, edgecolor="none", zorder=0))
    if board.is_check():
        x, y = xy(board.king(board.turn))
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor="#e81414", alpha=0.45, edgecolor="none", zorder=0.5))
    for sq, piece in board.piece_map().items():
        x, y = xy(sq)
        white = piece.color == chess.WHITE
        ax.text(x + 0.5, y + 0.45, GLYPH[piece.piece_type], ha="center", va="center", fontsize=26,
                family="DejaVu Sans", color="#fafafa" if white else "#141414", zorder=2,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="#1b1b1b")] if white else None)
    for m, w in moves_weights:
        (x0, y0), (x1, y1) = xy(m.from_square), xy(m.to_square)
        ax.add_patch(FancyArrow(x0 + .5, y0 + .5, x1 - x0, y1 - y0, width=0.06 + 0.14 * w, head_width=0.28 + 0.15 * w,
                                head_length=0.3, length_includes_head=True, color="#15781b",
                                alpha=0.35 + 0.55 * w, zorder=3, linewidth=0))
    files = "abcdefgh"
    for i in range(8):
        ax.text(i + 0.5, -0.25, files[7 - i] if flipped else files[i], ha="center", va="center", fontsize=9)
        ax.text(-0.25, i + 0.5, str(8 - i if flipped else i + 1), ha="center", va="center", fontsize=9)
    ax.set_xlim(-0.45, 8); ax.set_ylim(-0.45, 8); ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title)


def plot_board_and_policy(board: chess.Board, analysis: dict, top_k: int = 10, title: str | None = None):
    """Position (with arrows for the top-3 searched moves) next to visit share vs network prior.

    `analysis` is the dict returned by `mcts.analyse` (moves sorted by visits)."""
    import matplotlib.pyplot as plt
    moves = analysis["moves"][:top_k]
    fig, (axb, axp) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.0, 1.05]})
    top_share = max((m["share"] for m in moves[:3]), default=1.0) or 1.0
    draw_board(axb, board, [(m["move"], m["share"] / top_share) for m in moves[:3]],
               flipped=board.turn == chess.BLACK)
    y = np.arange(len(moves))[::-1]
    axp.barh(y + 0.2, [m["share"] for m in moves], height=0.4, color="#2a9d8f", label="MCTS visit share  N/ΣN")
    axp.barh(y - 0.2, [m["prior"] for m in moves], height=0.4, color="#e9c46a", label="network prior  P")
    for yi, m in zip(y, moves):
        axp.text(max(m["share"], m["prior"]) + 0.01, yi, f"Q={m['q']:+.2f}  N={m['visits']}", va="center", fontsize=8)
    axp.set_yticks(y, [m["san"] for m in moves]); axp.set_xlim(0, 1.15)
    axp.set_xlabel("probability"); axp.legend(loc="lower right", fontsize=9)
    side = "white" if board.turn == chess.WHITE else "black"
    axp.set_title(f"{side} to move — network value {analysis['value']:+.2f}, "
                  f"{sum(m['visits'] for m in analysis['moves'])} sims")
    fig.suptitle(title or "MCTS analysis"); fig.tight_layout()
    return fig


def plot_mcts_tree(root, max_depth: int = 2, top_k: int = 3, ax=None, title: str = "MCTS search tree"):
    """The search tree, top-`top_k` most visited children per node down to `max_depth`. Edges show the
    move (SAN), visit count N and mean value Q (from the mover's perspective); width ~ N."""
    import matplotlib.pyplot as plt
    pos, edges, x = {}, [], [0.0]

    def layout(node, depth):
        kids = []
        if depth < max_depth and not node.terminal and node.N.size:
            for i in np.argsort(-node.N)[:top_k]:
                if node.N[i] > 0 and node.children[i] is not None:
                    kids.append((int(i), node.children[i]))
        if kids:
            xs = []
            for i, ch in kids:
                layout(ch, depth + 1)
                xs.append(pos[id(ch)][0]); edges.append((node, ch, i))
            cx = sum(xs) / len(xs)
        else:
            cx = x[0]; x[0] += 1.0
        pos[id(node)] = (cx, -float(depth), node)
    layout(root, 0)
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(7.0, x[0] * 1.5), 2.2 + 1.9 * max_depth))
    maxN = max((float(p.N[i]) for p, _, i in edges), default=1.0)
    for parent, child, i in edges:
        x0, y0, _ = pos[id(parent)]; x1, y1, _ = pos[id(child)]
        n, q = float(parent.N[i]), float(parent.Q[i])
        ax.plot([x0, x1], [y0, y1], color="#9aa7b1", lw=1.0 + 6.0 * n / maxN, zorder=1)
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, f"{parent.board.san(parent.moves[i])}\nN={int(n)}\nQ={q:+.2f}",
                fontsize=8, ha="center", va="center", zorder=3,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#d0d0d0", alpha=0.95))
    for px, py, nd in pos.values():
        face = "#f4d03f" if nd.terminal else ("#2a9d8f" if nd is root else "#e76f51")
        ax.scatter([px], [py], s=420 if nd is root else 240, c=face, edgecolors="#15455f", lw=1.5, zorder=2)
    rx, ry, _ = pos[id(root)]
    ax.text(rx, ry + 0.22, "root", fontsize=9, ha="center", va="bottom")
    ax.set_axis_off(); ax.set_title(title); ax.margins(0.12)
    if fig is not None:
        fig.tight_layout()
        return fig
    return ax


# --------------------------------------------------------------------------- training curves
def load_run(run_dir) -> tuple[list[dict], list[dict]]:
    run_dir = Path(run_dir)
    read = lambda p: [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
    return read(run_dir / "log.jsonl"), read(run_dir / "eval.jsonl")


def plot_training(run_dir, baselines: dict | None = None):
    """Six panels of training progress. `baselines` = oracle['baselines'] adds reference lines."""
    import matplotlib.pyplot as plt
    log, ev = load_run(run_dir)
    if not log:
        raise ValueError(f"no log at {run_dir}")
    t = np.array([r["time_h"] for r in log])
    g = lambda k: np.array([r.get(k, np.nan) for r in log], dtype=float)
    fig, axs = plt.subplots(2, 3, figsize=(17, 9))
    ax = axs[0, 0]
    ax.plot(t, g("policy_ce"), label="policy CE"); ax.plot(t, g("policy_kl"), label="policy KL(π‖p)")
    ax.plot(t, g("value_mse"), label="value MSE"); ax.set_title("training loss"); ax.legend(); ax.set_xlabel("hours")
    ref_style = {"random": ":", "greedy": "-.", "minimax2": "--", "minimax3": (0, (5, 1))}
    for ax, metrics, ttl in [(axs[0, 1], ["top1", "good", "blunder"], "oracle: raw policy's move vs Stockfish"),
                             (axs[0, 2], ["accuracy", "wp_loss"], "oracle: Lichess accuracy / win-% lost per move")]:
        for k in metrics:
            line, = ax.plot(t, g(f"oracle/{k}"), label=k)
            for b, ls in ref_style.items():
                if baselines and k in baselines.get(b, {}):
                    ax.axhline(baselines[b][k], color=line.get_color(), ls=ls, lw=0.9, alpha=0.6)
        if baselines:
            for b, ls in ref_style.items():
                ax.plot([], [], color="gray", ls=ls, label=f"{b} (ref)")
        ax.set_title(ttl); ax.legend(fontsize=8); ax.set_xlabel("hours")
    ax = axs[1, 0]
    by_opp = {}
    for r in ev:
        by_opp.setdefault(r["opponent"], []).append((r["time_h"], r["score"], r["n"]))
    for opp, rs in by_opp.items():
        rs = np.array(rs)
        k = min(3, len(rs))                                    # smooth over the last few matches
        sm = np.convolve(rs[:, 1], np.ones(k) / k, mode="valid")
        ax.plot(rs[k - 1:, 0], sm, marker="o", ms=3, label=opp)
    ax.axhline(0.5, color="gray", lw=0.8, ls="--"); ax.set_ylim(-0.03, 1.03)
    ax.set_title("ladder: agent score vs opponents (MCTS, no noise)"); ax.legend(fontsize=8); ax.set_xlabel("hours")
    ax = axs[1, 1]
    ax.stackplot(t, g("sp/white"), g("sp/draw"), g("sp/black"), labels=["white wins", "draw", "black wins"],
                 colors=["#f1faee", "#a8dadc", "#457b9d"], edgecolor="#555", lw=0.3)
    ax2 = ax.twinx(); ax2.plot(t, g("sp/plies"), color="#e63946", label="game length"); ax2.set_ylabel("plies")
    ax.plot(t, np.nan_to_num(g("sp_term/checkmate")), color="k", lw=1.5, label="checkmate fraction")
    ax.set_title("self-play outcomes"); ax.legend(loc="upper left", fontsize=8); ax.set_xlabel("hours")
    ax = axs[1, 2]
    ax.plot(t, g("selfplay_plies_per_s"), label="self-play plies/s")
    ax.plot(t, g("infer_pos_per_s") / 100, label="network evals/s ÷ 100")
    ax.set_title("throughput"); ax.legend(fontsize=8); ax.set_xlabel("hours")
    fig.suptitle(f"AlphaZero chess — {Path(run_dir).name} (gen {log[-1]['gen']}, {log[-1]['games']} games)")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- game viewer
def _frames(moves: list[str], start_fen: str | None):
    b = chess.Board(start_fen) if start_fen else chess.Board()
    fens, sans, last, checks = [b.fen()], [], [None], [None]
    for u in moves:
        m = chess.Move.from_uci(u)
        sans.append(b.san(m))
        b.push(m)
        fens.append(b.fen()); last.append([u[:2], u[2:4]])
        checks.append(chess.square_name(b.king(b.turn)) if b.is_check() else None)
    return fens, sans, last, checks, b


def game_viewer_html(games: list[dict], path=None, title: str = "AlphaZero chess games") -> str:
    """Self-contained HTML replay viewer. Each game: {moves: [uci], start_fen?, headers?: {..},
    values?: [(ply, value_for_white)], flipped?: bool}. Writes to `path` if given; returns the HTML."""
    from web_board import BOARD_CSS, BOARD_JS
    data = []
    for g in games:
        fens, sans, last, checks, final = _frames(g["moves"], g.get("start_fen"))
        hdr = dict(g.get("headers", {}))
        hdr.setdefault("Result", final.result(claim_draw=True))
        data.append({"fens": fens, "sans": sans, "last": last, "checks": checks, "headers": hdr,
                     "values": g.get("values", []), "flipped": bool(g.get("flipped", False))})
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>{BOARD_CSS}
body{{margin:0;background:#161512;color:#e8e6e3;font:14px system-ui,sans-serif}}
.wrap{{display:flex;gap:22px;padding:18px;flex-wrap:wrap;align-items:flex-start}}
.side{{width:330px}} select,button{{background:#2b2a27;color:#e8e6e3;border:1px solid #444;border-radius:6px;padding:6px 9px;font-size:14px}}
button{{cursor:pointer;min-width:42px}} button:hover{{background:#3a3935}}
.moves{{height:300px;overflow-y:auto;background:#22211e;border-radius:6px;padding:6px;margin-top:10px;line-height:1.9}}
.mv{{padding:2px 5px;border-radius:4px;cursor:pointer}} .mv:hover{{background:#3a3935}} .mv.cur{{background:#5b8a32;color:#fff}}
.hdr{{color:#aaa;margin:8px 0;white-space:pre-line}} h3{{margin:4px 0 8px}} #graph{{margin-top:10px;background:#22211e;border-radius:6px}}
</style></head><body><div class="wrap">
<div id="board" style="--cb-size:min(560px,92vw)"></div>
<div class="side"><h3>{html.escape(title)}</h3>
<select id="sel" style="width:100%"></select><div class="hdr" id="hdr"></div>
<div><button onclick="go(0)">⏮</button> <button onclick="go(cur-1)">◀</button> <button id="pl" onclick="toggle()">▶ play</button>
<button onclick="go(cur+1)">▶</button> <button onclick="go(1e9)">⏭</button> <button onclick="flip=!flip;draw()">⇅</button></div>
<svg id="graph" width="330" height="90"></svg><div class="moves" id="moves"></div></div></div>
<script>{BOARD_JS}
const GAMES = {json.dumps(data)};
let gi=0, cur=0, flip=false, timer=null;
const sel=document.getElementById('sel');
GAMES.forEach((g,i)=>{{const o=document.createElement('option'); o.value=i;
  o.textContent=`${{i+1}}. ${{g.headers.White||'?'}} vs ${{g.headers.Black||'?'}}  ${{g.headers.Result||''}}`; sel.appendChild(o);}});
sel.onchange=()=>{{gi=+sel.value; cur=0; flip=GAMES[gi].flipped; build(); draw();}};
function build(){{
  const g=GAMES[gi], m=document.getElementById('moves'); m.innerHTML='';
  g.sans.forEach((s,i)=>{{ if(i%2===0) m.insertAdjacentHTML('beforeend', `<b>${{i/2+1}}.</b> `);
    const sp=document.createElement('span'); sp.className='mv'; sp.textContent=s; sp.onclick=()=>go(i+1); sp.id='mv'+(i+1); m.appendChild(sp); m.append(' '); }});
  document.getElementById('hdr').textContent = Object.entries(g.headers).map(([k,v])=>`${{k}}: ${{v}}`).join('\\n');
  graph();
}}
function graph(){{
  const g=GAMES[gi], s=document.getElementById('graph'), W=330, H=90, n=Math.max(g.sans.length,1);
  let p=`<line x1="0" y1="${{H/2}}" x2="${{W}}" y2="${{H/2}}" stroke="#555"/>`;
  if(g.values.length){{ p+=`<polyline fill="none" stroke="#7fb4e0" stroke-width="1.6" points="${{g.values.map(([t,v])=>`${{t/n*W}},${{H/2-v*H/2*0.9}}`).join(' ')}}"/>`;
    p+=`<text x="4" y="12" fill="#888" font-size="10">agent value (white's view)</text>`; }}
  p+=`<line id="gcur" x1="0" y1="0" x2="0" y2="${{H}}" stroke="#5b8a32"/>`; s.innerHTML=p;
}}
function draw(){{
  const g=GAMES[gi]; cur=Math.max(0,Math.min(cur,g.fens.length-1));
  const l=g.last[cur];
  cbRender(document.getElementById('board'), g.fens[cur], {{flipped:flip, lastMove:l||[], check:g.checks[cur]}});
  document.querySelectorAll('.mv').forEach(e=>e.classList.remove('cur'));
  const e=document.getElementById('mv'+cur); if(e){{e.classList.add('cur'); e.scrollIntoView({{block:'nearest'}});}}
  const gc=document.getElementById('gcur'); if(gc){{const x=cur/Math.max(g.sans.length,1)*330; gc.setAttribute('x1',x); gc.setAttribute('x2',x);}}
}}
function go(i){{cur=i; draw();}}
function toggle(){{ if(timer){{clearInterval(timer); timer=null; document.getElementById('pl').textContent='▶ play';}}
  else {{timer=setInterval(()=>{{ if(cur>=GAMES[gi].fens.length-1) toggle(); else go(cur+1); }}, 600); document.getElementById('pl').textContent='⏸ pause';}} }}
document.addEventListener('keydown',e=>{{ if(e.key==='ArrowLeft') go(cur-1); if(e.key==='ArrowRight') go(cur+1); }});
flip=GAMES.length?GAMES[0].flipped:false; build(); draw();
</script></body></html>"""
    if path is not None:
        Path(path).write_text(page)
    return page


def pgn_to_games(pgn_path, limit: int | None = None, agent_name_prefix: str = "agent") -> list[dict]:
    """Read games from a PGN file (e.g. runs/<run>/eval_games.pgn) into `game_viewer_html` format;
    games where the agent plays black are shown flipped."""
    import chess.pgn
    out = []
    with open(pgn_path) as f:
        while limit is None or len(out) < limit:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            out.append({"moves": [m.uci() for m in g.mainline_moves()], "start_fen": g.board().fen(),
                        "headers": dict(g.headers),
                        "flipped": g.headers.get("Black", "").startswith(agent_name_prefix)})
    return out


if __name__ == "__main__":
    # python viz.py runs/r1  ->  runs/r1/training.png + runs/r1/eval_viewer.html (latest ladder games)
    import sys
    import matplotlib
    matplotlib.use("Agg")
    run = Path(sys.argv[1])
    baselines = None
    try:
        from oracle import DEFAULT_PATH, load_oracle
        baselines = load_oracle(DEFAULT_PATH)["baselines"] if DEFAULT_PATH.exists() else None
    except Exception:
        pass
    plot_training(run, baselines).savefig(run / "training.png", dpi=90)
    games = pgn_to_games(run / "eval_games.pgn") if (run / "eval_games.pgn").exists() else []
    game_viewer_html(games[-48:], run / "eval_viewer.html", title=f"{run.name}: latest ladder games")
    log, ev = load_run(run)
    last = log[-1]
    print(f"gen {last['gen']}  {last['time_h']:.2f}h  {last['games']} games  oracle top1 {last.get('oracle/top1', float('nan')):.3f} "
          f"acc {last.get('oracle/accuracy', float('nan')):.1f}")
    latest = {}
    for r in ev:
        latest.setdefault(r["opponent"], []).append(r)
    for opp, rs in latest.items():
        print(f"  vs {opp:<11} latest score {rs[-1]['score']:.2f} (gen {rs[-1]['gen']}, +{rs[-1]['wins']} ={rs[-1]['draws']} -{rs[-1]['losses']})")
    print(f"wrote {run / 'training.png'} and {run / 'eval_viewer.html'}")
