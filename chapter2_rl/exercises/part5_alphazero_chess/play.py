"""
Play against (or watch) the trained agent in the browser.

    python play.py --ckpt runs/r1/latest.pt            # then open http://localhost:8080
    python play.py --ckpt runs/r1/latest.pt --port 8081 --sims 400

Modes:
  * play:  you vs the agent (click a piece, then a highlighted square; promotions use the chosen piece)
  * watch: the agent vs any opponent from the ladder (random, greedy, minimax2/3, Stockfish levels),
           stepping one move at a time or on autoplay
The side panel shows the agent's last search (top moves: visit share, prior, Q, also drawn as arrows on
the board), its value estimate over the game, and optionally Stockfish's evaluation for comparison.
"""
from __future__ import annotations

import argparse
import http.server
import json
import threading

import chess
import chess.engine
import torch

import chapter2_rl.exercises.part5_alphazero_chess.mcts as M
from chapter2_rl.exercises.part5_alphazero_chess.model import load_model
from chapter2_rl.exercises.part5_alphazero_chess.opponents import OPPONENTS, STOCKFISH, make_opponent
from chapter2_rl.exercises.part5_alphazero_chess.web_board import BOARD_CSS, BOARD_JS


class Session:
    def __init__(self, evaluator, sims: int, c_puct: float):
        self.ev, self.default_sims, self.c_puct = evaluator, sims, c_puct
        self.lock = threading.Lock()
        self.sf = None
        self.opponent = None
        self.new({"mode": "play", "human": "white", "sims": sims})

    # -- helpers
    def _sf_eval(self, board):
        if not self.show_sf or board.is_game_over(claim_draw=True):
            return None
        if self.sf is None:
            self.sf = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
        info = self.sf.analyse(board, chess.engine.Limit(depth=14))
        return {"cp": info["score"].white().score(mate_score=10000), "best": info["pv"][0].uci() if info.get("pv") else None,
                "mate": info["score"].white().mate()}

    def _agent_move(self):
        a = M.analyse(self.board, self.ev, self.sims, self.c_puct)
        sign = 1 if self.board.turn == chess.WHITE else -1
        self.values.append([len(self.board.move_stack), sign * a["value"]])
        self.analysis = {"value": a["value"], "side": "white" if self.board.turn == chess.WHITE else "black",
                         "moves": [{k: m[k] for k in ("san", "uci", "visits", "share", "prior", "q")} for m in a["moves"][:10]]}
        self._push(a["best"])

    def _push(self, mv):
        self.sans.append(self.board.san(mv))
        self.board.push(mv)
        e = self._sf_eval(self.board)
        if e is not None:
            self.sf_hist.append([len(self.board.move_stack), e["cp"]])
        self.sf_now = e

    def agent_to_move(self):
        if self.board.is_game_over(claim_draw=True):
            return False
        white = self.board.turn == chess.WHITE
        return (white == (self.agent_color == "white")) or self.agent_color == "both"

    # -- actions
    def new(self, p):
        self.mode = p.get("mode", "play")
        self.sims = int(p.get("sims", self.default_sims))
        self.show_sf = bool(p.get("show_sf", False))
        fen = p.get("fen") or chess.STARTING_FEN
        self.board = chess.Board(fen)
        self.sans, self.values, self.sf_hist, self.analysis, self.sf_now = [], [], [], None, None
        if self.opponent is not None:
            self.opponent.close(); self.opponent = None
        if self.mode == "play":
            self.agent_color = "black" if p.get("human", "white") == "white" else "white"
        else:
            self.agent_color = p.get("agent", "white")
            opp = p.get("opponent", "minimax2")
            if opp != "agent":
                self.opponent = make_opponent(opp)
            else:
                self.agent_color = "both"
            self.opp_name = opp
        self.sf_now = self._sf_eval(self.board)
        if self.mode == "play" and self.agent_to_move():
            self._agent_move()

    def human_move(self, uci):
        mv = chess.Move.from_uci(uci)
        if mv not in self.board.legal_moves:
            return "illegal move"
        self._push(mv)
        if self.agent_to_move():
            self._agent_move()

    def step(self):
        if self.board.is_game_over(claim_draw=True):
            return
        if self.agent_to_move():
            self._agent_move()
        else:
            self._push(self.opponent.move(self.board))

    def undo(self):
        n = 2 if self.mode == "play" else 1
        for _ in range(min(n, len(self.board.move_stack))):
            self.board.pop(); self.sans.pop()
            ply = len(self.board.move_stack)
            self.values = [v for v in self.values if v[0] < ply]
            self.sf_hist = [v for v in self.sf_hist if v[0] <= ply]
        self.analysis = None

    def state(self):
        b = self.board
        over = b.is_game_over(claim_draw=True)
        res = None
        if over:
            o = b.outcome(claim_draw=True)
            res = f"{o.result()} ({o.termination.name.lower().replace('_', ' ')})"
        legal = [{"from": chess.square_name(m.from_square), "to": chess.square_name(m.to_square), "uci": m.uci(),
                  "promo": bool(m.promotion)} for m in b.legal_moves] if not over else []
        last = [chess.square_name(b.peek().from_square), chess.square_name(b.peek().to_square)] if b.move_stack else []
        return {"fen": b.fen(), "turn": "white" if b.turn == chess.WHITE else "black", "legal": legal, "last": last,
                "check": chess.square_name(b.king(b.turn)) if b.is_check() else None, "sans": self.sans,
                "result": res, "mode": self.mode, "agent_color": self.agent_color, "analysis": self.analysis,
                "values": self.values, "sf_hist": self.sf_hist, "sf_now": self.sf_now, "sims": self.sims,
                "agent_to_move": self.agent_to_move(), "show_sf": self.show_sf, "ply": len(b.move_stack)}


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>AlphaZero chess</title><style>%CSS%
body{margin:0;background:#161512;color:#e8e6e3;font:14px system-ui,sans-serif}
.wrap{display:flex;gap:24px;padding:18px;flex-wrap:wrap;align-items:flex-start}
.panel{width:380px} .row{margin:6px 0;display:flex;gap:6px;align-items:center;flex-wrap:wrap}
select,button,input{background:#2b2a27;color:#e8e6e3;border:1px solid #444;border-radius:6px;padding:6px 9px;font-size:14px}
button{cursor:pointer} button:hover{background:#3a3935} button.primary{background:#5b8a32;border-color:#5b8a32;color:#fff}
#status{font-size:17px;margin:10px 0;min-height:22px} .card{background:#22211e;border-radius:8px;padding:10px;margin-top:10px}
table{width:100%;border-collapse:collapse;font-size:13px} td{padding:2px 4px} .bar{height:9px;border-radius:3px;background:#2a9d8f}
.bar.p{background:#e9c46a} .moves{max-height:150px;overflow-y:auto;line-height:1.8} .muted{color:#999}
</style></head><body><div class="wrap">
<div id="board" style="--cb-size:min(600px,94vw)"></div>
<div class="panel">
 <div class="row"><select id="mode"><option value="play">play vs agent</option><option value="watch">watch agent vs …</option></select>
  <select id="human"><option value="white">you: white</option><option value="black">you: black</option></select>
  <select id="agentc" style="display:none"><option value="white">agent: white</option><option value="black">agent: black</option></select>
  <select id="opp" style="display:none">%OPPS%<option value="agent">agent (self-play)</option></select></div>
 <div class="row">sims <input id="sims" type="number" value="%SIMS%" min="1" max="5000" style="width:80px">
  <label><input type="checkbox" id="sf"> Stockfish eval</label>
  promote <select id="promo"><option value="q">Q</option><option value="r">R</option><option value="b">B</option><option value="n">N</option></select></div>
 <div class="row"><button class="primary" onclick="newGame()">new game</button><button onclick="post('/api/undo')">undo</button>
  <button id="stepb" onclick="post('/api/step')" style="display:none">step</button>
  <button id="autob" onclick="auto()" style="display:none">▶ autoplay</button><button onclick="flip=!flip;draw()">⇅ flip</button></div>
 <div id="status"></div>
 <div class="card"><b>agent's last search</b> <span class="muted" id="aval"></span>
   <table id="an"></table></div>
 <div class="card"><b>evaluation</b> <span class="muted">(white's view: agent value, Stockfish expected score)</span>
   <svg id="graph" width="360" height="110"></svg></div>
 <div class="card moves" id="moves"></div>
</div></div>
<script>%JS%
let S=null, sel=null, flip=false, timer=null, busy=false;
const $=id=>document.getElementById(id);
$('mode').onchange=()=>{const w=$('mode').value==='watch'; $('human').style.display=w?'none':''; ['agentc','opp','stepb','autob'].forEach(i=>$(i).style.display=w?'':'none');};
async function post(url, body={}){ if(busy) return; busy=true; $('status').textContent='thinking…';
  try{ const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}); S=await r.json(); }
  finally{ busy=false; } sel=null; draw(); }
function newGame(){ if(timer) auto(); flip = $('mode').value==='play' ? $('human').value==='black' : $('agentc').value==='black';
  post('/api/new',{mode:$('mode').value, human:$('human').value, agent:$('agentc').value, opponent:$('opp').value, sims:+$('sims').value, show_sf:$('sf').checked}); }
function auto(){ if(timer){clearInterval(timer); timer=null; $('autob').textContent='▶ autoplay'; return;}
  $('autob').textContent='⏸ pause'; timer=setInterval(()=>{ if(!busy){ if(S && S.result) auto(); else post('/api/step'); } }, 250); }
function click(sq){
  if(!S || S.result || S.mode!=='play' || S.agent_to_move) return;
  const mine = S.legal.filter(m=>m.from===sq);
  if(sel){ const mv=S.legal.filter(m=>m.from===sel && m.to===sq);
    if(mv.length){ let u=mv[0].uci; if(mv[0].promo) u=u.slice(0,4)+$('promo').value; post('/api/move',{uci:u}); return; } }
  sel = mine.length ? sq : null; draw();
}
function draw(){
  if(!S) return;
  // green: the agent's top-3 searched moves (width ~ visit share); blue: Stockfish's best move
  const arrows = S.analysis ? S.analysis.moves.slice(0,3).map(m=>({from:m.uci.slice(0,2), to:m.uci.slice(2,4),
     w:m.share/S.analysis.moves[0].share, color:'#15781b'})) : [];
  if(S.sf_now && S.sf_now.best) arrows.push({from:S.sf_now.best.slice(0,2), to:S.sf_now.best.slice(2,4), w:.55, color:'#1f5fbf'});
  const targets = sel ? S.legal.filter(m=>m.from===sel).map(m=>m.to) : [];
  cbRender($('board'), S.fen, {flipped:flip, lastMove:S.last, check:S.check, selected:sel, targets, arrows:S.result?[]:arrows, onClick:click});
  let st = S.result ? 'game over: '+S.result : (S.turn+' to move' + (S.mode==='play' ? (S.agent_to_move?' (agent)':' (you)') : ''));
  if(S.sf_now) st += S.sf_now.mate!=null ? `  ·  SF: mate ${S.sf_now.mate}` : `  ·  SF: ${(S.sf_now.cp/100).toFixed(2)}`;
  $('status').textContent = st;
  const a=S.analysis; $('aval').textContent = a ? `value ${a.value>=0?'+':''}${a.value.toFixed(2)} for ${a.side} · ${S.sims} sims` : '';
  $('an').innerHTML = a ? '<tr class="muted"><td>move</td><td>visits</td><td></td><td>prior</td><td></td><td>Q</td></tr>' + a.moves.map(m=>
    `<tr><td>${m.san}</td><td>${(100*m.share).toFixed(0)}%</td><td style="width:70px"><div class="bar" style="width:${100*m.share}%"></div></td>
     <td>${(100*m.prior).toFixed(0)}%</td><td style="width:60px"><div class="bar p" style="width:${100*m.prior}%"></div></td><td>${m.q>=0?'+':''}${m.q.toFixed(2)}</td></tr>`).join('') : '';
  const W=360,H=110,n=Math.max(S.ply,10), y=v=>H/2-v*H/2*0.92, sfv=cp=>2/(1+Math.exp(-0.00368208*cp))-1;
  let g=`<line x1="0" y1="${H/2}" x2="${W}" y2="${H/2}" stroke="#555"/>`;
  if(S.values.length) g+=`<polyline fill="none" stroke="#7fd36b" stroke-width="1.8" points="${S.values.map(([t,v])=>`${t/n*W},${y(v)}`).join(' ')}"/>`;
  if(S.sf_hist.length) g+=`<polyline fill="none" stroke="#6aa0ff" stroke-width="1.4" stroke-dasharray="4 2" points="${S.sf_hist.map(([t,c])=>`${t/n*W},${y(sfv(c))}`).join(' ')}"/>`;
  g+=`<text x="4" y="12" fill="#7fd36b" font-size="10">agent</text><text x="44" y="12" fill="#6aa0ff" font-size="10">stockfish</text>`;
  $('graph').innerHTML=g;
  $('moves').innerHTML = S.sans.map((s,i)=>(i%2===0?`<b>${i/2+1}.</b> `:'')+s).join(' ');
  $('moves').scrollTop = 1e9;
}
post('/api/state');
</script></body></html>"""


def serve(session: Session, port: int):
    page = (PAGE.replace("%CSS%", BOARD_CSS).replace("%JS%", BOARD_JS).replace("%SIMS%", str(session.default_sims))
            .replace("%OPPS%", "".join(f'<option value="{o}">{o}</option>' for o in OPPONENTS + ["sf_d8", "sf_full"])))

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, code, body: bytes, ctype):
            self.send_response(code); self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            self._send(200, page.encode(), "text/html; charset=utf-8")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            p = json.loads(self.rfile.read(n) or b"{}")
            with session.lock:
                try:
                    if self.path == "/api/new":
                        session.new(p)
                    elif self.path == "/api/move":
                        session.human_move(p["uci"])
                    elif self.path == "/api/step":
                        session.step()
                    elif self.path == "/api/undo":
                        session.undo()
                    body = session.state()
                except Exception as e:                       # keep the server alive; report in the status line
                    body = {**session.state(), "result": f"error: {e}"}
            self._send(200, json.dumps(body).encode(), "application/json")

        def log_message(self, *a):
            pass

    with http.server.HTTPServer(("0.0.0.0", port), Handler) as httpd:
        print(f"▶ open http://localhost:{port}   (Ctrl-C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")
        finally:
            if session.sf is not None:
                session.sf.quit()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--c-puct", type=float, default=2.0)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(a.ckpt, dev)
    print(f"loaded {a.ckpt} (gen {ckpt.get('gen', '?')}, {ckpt.get('games', '?')} self-play games)")
    serve(Session(M.LocalEvaluator(model), a.sims, a.c_puct), a.port)
