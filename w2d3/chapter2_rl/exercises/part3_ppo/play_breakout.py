#!/usr/bin/env python3
"""Play the VM's Atari Breakout from a browser, without a desktop display.

VM:    /opt/arena-env/bin/python play_breakout.py
Local: ssh -N -L 8765:127.0.0.1:8765 arena9-coco
Then open http://127.0.0.1:8765. Use --desktop for the original pygame mode.
"""

import argparse
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import ale_py
import gymnasium as gym


HTML = r"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Breakout · ARENA</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#11131b;color:#ebedf5;font:16px system-ui;display:grid;place-items:center;min-height:100vh}
main{width:min(94vw,560px);padding:24px}header{display:flex;justify-content:space-between;align-items:baseline}h1{margin:0;font-size:26px}#status{color:#aeb8cc;font-size:14px}
#screen{display:block;width:min(100%,480px);height:auto;margin:20px auto;background:#000;image-rendering:pixelated;border:1px solid #343b50;outline:none}
section{display:flex;gap:10px;flex-wrap:wrap}button{font:inherit;border:1px solid #4b5571;border-radius:8px;background:#222a3e;color:inherit;padding:10px 18px;cursor:pointer;touch-action:none}button:active{background:#41527b}
p{color:#aeb8cc;line-height:1.6;font-size:14px}#message{color:#ffc674;min-height:22px}
</style><main><header><h1>Breakout</h1><span id="status">Connecting…</span></header>
<img id="screen" src="/stream" width="160" height="210" tabindex="0" alt="Breakout game; use arrow keys and space">
<section><button data-key="left">← Left</button><button data-key="fire">Fire</button><button data-key="right">Right →</button><button id="pause">Resume</button><button id="reset">Reset</button></section>
<p>← / → or A / D to move · Space / F to fire · P to pause · R to reset.<br>Click the game to focus. Hold Fire to launch again after losing a life.</p><div id="message"></div></main>
<script>
const screen=document.querySelector('#screen'),status=document.querySelector('#status'),message=document.querySelector('#message');
let keys={left:false,right:false,fire:false},paused=true;
const mapping={ArrowLeft:'left',KeyA:'left',ArrowRight:'right',KeyD:'right',Space:'fire',KeyF:'fire'};
async function send(extra={}){try{const r=await fetch('/input',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...keys,...extra})});if(!r.ok)throw Error(await r.text());}catch(e){message.textContent='Connection lost: '+e.message;}}
function clear(){keys={left:false,right:false,fire:false};send({pause:true});}
window.addEventListener('keydown',e=>{if(mapping[e.code]){e.preventDefault();keys[mapping[e.code]]=true;send();}else if(!e.repeat&&e.code==='KeyP'){e.preventDefault();send({pause:!paused});}else if(!e.repeat&&e.code==='KeyR'){send({reset:true});}});
window.addEventListener('keyup',e=>{if(mapping[e.code]){e.preventDefault();keys[mapping[e.code]]=false;send();}});
window.addEventListener('blur',clear);document.addEventListener('visibilitychange',()=>{if(document.hidden)clear();});
document.querySelectorAll('[data-key]').forEach(b=>{b.onpointerdown=e=>{e.preventDefault();b.setPointerCapture(e.pointerId);keys[b.dataset.key]=true;send();};b.onpointerup=b.onpointercancel=()=>{keys[b.dataset.key]=false;send();};});
document.querySelector('#pause').onclick=()=>send({pause:!paused});document.querySelector('#reset').onclick=()=>send({reset:true});screen.onclick=()=>screen.focus();
async function refresh(){try{const r=await fetch('/state',{cache:'no-store'});if(!r.ok)throw Error(await r.text());const s=await r.json();paused=s.paused;status.textContent=`Score ${s.score} · Lives ${s.lives}`;document.querySelector('#pause').textContent=paused?'Resume':'Pause';message.textContent=s.game_over?'Game over — press R to restart.':paused?'Paused — click Resume or press P.':'';}catch(e){message.textContent='Connection lost: '+e.message;}finally{setTimeout(refresh,250);}}
setInterval(()=>{if(!document.hidden)send();},250);refresh();screen.focus();
</script></html>"""


def make_env(seed):
    gym.register_envs(ale_py)
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array", frameskip=1,
                   repeat_action_probability=0.0, full_action_space=True)
    env.reset(seed=seed)
    return env


class Game:
    def __init__(self, seed, fps):
        self.env = make_env(seed)
        self.actions = {name: i for i, name in enumerate(self.env.unwrapped.get_action_meanings())}
        self.fps = fps
        self.condition = threading.Condition()
        self.controls = dict(left=False, right=False, fire=False)
        self.last_input = time.monotonic()
        self.paused = True
        self.reset_requested = False
        self.game_over = False
        self.score = 0.0
        self.lives = self.env.unwrapped.ale.lives()
        self.frame_number = 0
        self.stopping = threading.Event()
        self.error = None
        self.encode_frame()

    def encode_frame(self):
        from PIL import Image
        out = io.BytesIO()
        Image.fromarray(self.env.render()).save(out, format="JPEG", quality=90, subsampling=0)
        self.frame = out.getvalue()
        self.frame_number += 1

    def input(self, data):
        with self.condition:
            for key in self.controls:
                self.controls[key] = bool(data.get(key, False))
            if "pause" in data:
                self.paused = bool(data["pause"])
            if data.get("reset"):
                self.reset_requested = True
            self.last_input = time.monotonic()

    def run(self):
        deadline = time.monotonic()
        try:
            while not self.stopping.is_set():
                with self.condition:
                    if self.reset_requested:
                        self.env.reset()
                        self.score = 0.0
                        self.game_over = False
                        self.reset_requested = False
                        self.lives = self.env.unwrapped.ale.lives()
                        self.encode_frame()
                        self.condition.notify_all()
                    if time.monotonic() - self.last_input > 1.5:
                        self.controls = dict(left=False, right=False, fire=False)
                        self.paused = True
                    if not self.paused and not self.game_over:
                        left, right, fire = (self.controls[k] for k in ("left", "right", "fire"))
                        action = "LEFT" if left and not right else "RIGHT" if right and not left else ""
                        action += "FIRE" if fire else ""
                        _, reward, terminated, truncated, info = self.env.step(self.actions[action or "NOOP"])
                        self.score += float(reward)
                        self.lives = int(info.get("lives", self.env.unwrapped.ale.lives()))
                        self.game_over = bool(terminated or truncated)
                        self.encode_frame()
                        self.condition.notify_all()
                deadline += 1 / self.fps
                delay = deadline - time.monotonic()
                if delay < -0.1:
                    deadline = time.monotonic()
                self.stopping.wait(max(0, delay))
        except Exception as exc:
            with self.condition:
                self.error = str(exc)
                self.condition.notify_all()
            raise
        finally:
            self.env.close()


def handler_for(game):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, body, content_type="application/json", status=200, frame=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if frame is not None:
                self.send_header("X-Frame", str(frame))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                return self.reply(HTML.encode(), "text/html; charset=utf-8")
            if path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                previous = -1
                try:
                    while not game.stopping.is_set():
                        with game.condition:
                            game.condition.wait_for(lambda: game.frame_number != previous or game.error, timeout=1)
                            if game.error:
                                return
                            frame, previous = game.frame, game.frame_number
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                            + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            with game.condition:
                if game.error:
                    return self.reply(game.error.encode(), "text/plain", 500)
                if path == "/state":
                    return self.reply(json.dumps(dict(score=game.score, lives=game.lives,
                        paused=game.paused, game_over=game.game_over, frame=game.frame_number)).encode())
                if path == "/frame":
                    from urllib.parse import parse_qs
                    try:
                        after = int(parse_qs(urlsplit(self.path).query).get("after", ["0"])[0])
                    except ValueError:
                        return self.reply(b"Invalid frame number", "text/plain", 400)
                    game.condition.wait_for(lambda: game.frame_number > after or game.error, timeout=0.1)
                    return self.reply(game.frame, "image/jpeg", frame=game.frame_number)
            self.reply(b"Not found", "text/plain", 404)

        def do_POST(self):
            if urlsplit(self.path).path != "/input":
                return self.reply(b"Not found", "text/plain", 404)
            # Inputs originate from this page. Reject cross-site requests to the forwarded port.
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                return self.reply(b"Invalid origin", "text/plain", 403)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024:
                    raise ValueError("Invalid input size")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("Expected an object")
                game.input(data)
            except (ValueError, TypeError) as exc:
                return self.reply(str(exc).encode(), "text/plain", 400)
            self.reply(b"{}")
    return Handler


def desktop(seed, fps):
    import pygame
    env = make_env(seed)
    pygame.init()
    screen = pygame.display.set_mode((480, 630))
    pygame.display.set_caption("Breakout · arrows/A/D, Space/F, R reset, Q quit")
    clock = pygame.time.Clock()
    actions = {name: i for i, name in enumerate(env.unwrapped.get_action_meanings())}
    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    env.reset()
            if not running:
                break
            keys = pygame.key.get_pressed()
            action = "LEFT" if keys[pygame.K_LEFT] or keys[pygame.K_a] else "RIGHT" if keys[pygame.K_RIGHT] or keys[pygame.K_d] else ""
            action += "FIRE" if keys[pygame.K_SPACE] or keys[pygame.K_f] else ""
            _, _, terminated, truncated, _ = env.step(actions[action or "NOOP"])
            frame = pygame.surfarray.make_surface(env.render().swapaxes(0, 1))
            screen.blit(pygame.transform.scale(frame, screen.get_size()), (0, 0))
            pygame.display.flip()
            if terminated or truncated:
                env.reset()
            clock.tick(fps)
    finally:
        env.close()
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--desktop", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.fps <= 120:
        parser.error("--fps must be between 1 and 120")
    if args.desktop:
        return desktop(args.seed, args.fps)
    game = Game(args.seed, args.fps)
    try:
        server = ThreadingHTTPServer((args.host, args.port), handler_for(game))
    except OSError:
        game.env.close()
        raise
    thread = threading.Thread(target=game.run, name="breakout", daemon=True)
    thread.start()
    print(f"Breakout ready at http://{args.host}:{args.port} (starts paused)", flush=True)
    print(f"Forward locally: ssh -N -L {args.port}:127.0.0.1:{args.port} arena9-coco", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        game.stopping.set()
        thread.join(timeout=3)


if __name__ == "__main__":
    main()
