#!/usr/bin/env python3
"""Mirror every run on the shared Jupyter kernel into a notebook file that VS Code shows live.

Start it next to the Jupyter server:   python kernel_mirror.py
Then in VS Code: drag ARENA_materials/shared_output.ipynb from the Explorer into a pane.

It follows the kernel of the "console-1" session with the most clients (or, failing that, any session's kernel with the most clients).
It only listens on the kernel's output channel and never sends the kernel anything.
"""
import argparse
import json
import os
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path
from queue import Empty
from urllib.request import Request, urlopen

from jupyter_client import BlockingKernelClient
from jupyter_core.paths import jupyter_runtime_dir
from jupyter_server.serverapp import list_running_servers

HEADER = "**Shared output**: every run on the shared kernel, newest first. Read-only (edits here get overwritten)."
WIDGET = "application/vnd.jupyter.widget-view+json"   # needs a live widget connection, so it can't be mirrored
STREAM_CAP = 200_000                                  # keep the last 200k characters of each printed output


def find_kernel():
    """Id of the shared kernel on the running Jupyter server, or None."""
    for srv in list_running_servers():
        req = Request(srv["url"].rstrip("/") + "/api/sessions", headers={"Authorization": f"token {srv['token']}"})
        try:
            sessions = json.load(urlopen(req, timeout=5))
        except OSError:
            continue
        if sessions:
            best = max(sessions, key=lambda s: ("console-1" in s["path"], s["kernel"]["connections"]))
            return best["kernel"]["id"]
    return None


def vscode_internal(code):
    """Setup code VS Code's Jupyter extension runs on the kernel by itself; not a cell anyone ran."""
    return ("_VSCODE_" in code
            or code.startswith("# Copyright (c) Microsoft Corporation")
            or re.fullmatch(r"(__file__|__vsc_ipynb_file__) = '[^\n]*'\s*", code) is not None
            or code.strip() == "%config Completer.use_jedi = False")


def repo_of(path):
    """(top folder, .git folder) of the git repo holding path, or None."""
    try:
        return subprocess.run(["git", "-C", str(path.parent), "rev-parse", "--show-toplevel", "--absolute-git-dir"],
                              capture_output=True, text=True, check=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return None


def keep_out_of_git(path, top, git_dir):
    """The repo gets auto-committed, so list the mirror file (and its temp file) in .git/info/exclude."""
    exclude = Path(git_dir) / "info" / "exclude"
    pattern = f"/{path.resolve().relative_to(top)}*"
    lines = exclude.read_text().splitlines() if exclude.exists() else []
    if pattern not in lines:
        exclude.parent.mkdir(exist_ok=True)
        exclude.write_text("".join(f"{l}\n" for l in lines + [pattern]))


def make_read_only(path, top):
    """Mark the mirror file read-only for VS Code in the repo's workspace settings.

    VS Code stops reloading a notebook from disk once it has unsaved changes (a stray keystroke, or the Jupyter
    extension picking a kernel for it), so without this the file can freeze for one person.
    """
    settings = Path(top) / ".vscode" / "settings.json"
    key = f"**/{path.name}"
    try:
        cfg = json.loads(settings.read_text()) if settings.exists() else {}
    except ValueError:
        print(f'could not parse {settings}; add "files.readonlyInclude": {{"{key}": true}} to it yourself', flush=True)
        return
    readonly = cfg.setdefault("files.readonlyInclude", {})
    if readonly.get(key) is not True:
        readonly[key] = True
        settings.parent.mkdir(exist_ok=True)
        settings.write_text(json.dumps(cfg, indent=2) + "\n")


def squash_cr(text):
    """Apply carriage returns the way a terminal does, so progress bars don't pile up."""
    lines = []
    for line in text.split("\n"):
        if "\r" in line:
            parts = [p for p in line.split("\r") if p]
            line = parts[-1] if parts else ""
        lines.append(line)
    return "\n".join(lines)


def markdown(text, cell_id=None):
    return {"id": cell_id or uuid.uuid4().hex[:8], "cell_type": "markdown", "metadata": {}, "source": text}


class Mirror:
    def __init__(self, path, max_cells, newest_first):
        self.path, self.max_cells, self.newest_first = path, max_cells, newest_first
        self.lock = threading.Lock()
        self.dirty = threading.Event()
        self.cells = self._load()
        self.by_parent = {}       # id of the request that ran a cell -> that cell
        self.displays = {}        # display_id -> outputs that update_display_data replaces
        self.clear_on_next = set()
        threading.Thread(target=self._writer, daemon=True).start()
        self.dirty.set()

    def _load(self):
        """Keep the history across restarts of this script (held oldest first in memory)."""
        try:
            nb = json.loads(self.path.read_text())
            cells = [c for c in nb["cells"] if c["id"] != "header"]
            return cells[::-1] if nb["metadata"].get("newest_first") else cells
        except (OSError, ValueError, KeyError):
            return []

    def handle(self, msg):
        kind, c = msg["msg_type"], msg["content"]
        parent = msg["parent_header"].get("msg_id")
        with self.lock:
            if kind == "execute_input":
                if vscode_internal(c["code"]):
                    return
                code = re.sub(r"\A# %%[^\n]*\n", "", c["code"])
                cell = {"id": uuid.uuid4().hex[:8], "cell_type": "code", "metadata": {},
                        "execution_count": c.get("execution_count"), "source": code, "outputs": []}
                self.cells.append(cell)
                del self.cells[:-self.max_cells]
                alive = {id(x) for x in self.cells}
                self.by_parent = {k: v for k, v in self.by_parent.items() if id(v) in alive}
                self.by_parent[parent] = cell
            elif kind == "status" and c["execution_state"] == "starting":
                self.cells.append(markdown("*kernel restarted*"))
            elif kind == "update_display_data":
                for out in self.displays.get(c.get("transient", {}).get("display_id"), []):
                    out["data"], out["metadata"] = c["data"], c.get("metadata", {})
            elif parent not in self.by_parent or not self._output(self.by_parent[parent], parent, kind, c):
                return
            self.dirty.set()

    def _output(self, cell, parent, kind, c):
        outs = cell["outputs"]
        if kind == "clear_output":
            if c.get("wait"):
                self.clear_on_next.add(parent)
            else:
                outs.clear()
            return True
        if kind not in ("stream", "display_data", "execute_result", "error"):
            return False
        if kind in ("display_data", "execute_result") and WIDGET in c["data"]:
            return False
        if parent in self.clear_on_next:
            self.clear_on_next.discard(parent)
            outs.clear()
        if kind == "stream":
            if outs and outs[-1]["output_type"] == "stream" and outs[-1]["name"] == c["name"]:
                outs[-1]["text"] = squash_cr(outs[-1]["text"] + c["text"])[-STREAM_CAP:]
            else:
                outs.append({"output_type": "stream", "name": c["name"], "text": squash_cr(c["text"])})
        elif kind == "error":
            outs.append({"output_type": "error", "ename": c["ename"], "evalue": c["evalue"], "traceback": c["traceback"]})
        else:
            out = {"output_type": kind, "data": c["data"], "metadata": c.get("metadata", {})}
            if kind == "execute_result":
                out["execution_count"] = c.get("execution_count")
            outs.append(out)
            display_id = c.get("transient", {}).get("display_id")
            if display_id:
                self.displays.setdefault(display_id, []).append(out)
        return True

    def _writer(self):
        """Rewrite the file after each burst of output; the rename makes every write atomic.

        VS Code reloads the open notebook but doesn't scroll to new cells, so by default the newest run
        goes at the top, where it is always in view.
        """
        while True:
            self.dirty.wait()
            time.sleep(0.2)
            self.dirty.clear()
            with self.lock:
                cells = self.cells[::-1] if self.newest_first else self.cells
                text = json.dumps({"cells": [markdown(HEADER if self.newest_first else HEADER.replace(", newest first", ""), "header")] + cells,
                                   "metadata": {"language_info": {"name": "python"}, "newest_first": self.newest_first},
                                   "nbformat": 4, "nbformat_minor": 5}, indent=1)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(text)
            os.replace(tmp, self.path)


def follow(kernel_id, mirror):
    """Copy the kernel's output channel into the mirror until the shared kernel changes."""
    cf = Path(jupyter_runtime_dir()) / f"kernel-{kernel_id}.json"
    if not cf.exists():
        time.sleep(3)
        return
    kc = BlockingKernelClient(connection_file=str(cf))
    kc.load_connection_file()
    kc.start_channels(shell=False, iopub=True, stdin=False, hb=False, control=False)
    checked = time.monotonic()
    try:
        while True:
            try:
                mirror.handle(kc.get_iopub_msg(timeout=1))
            except Empty:
                pass
            if time.monotonic() - checked > 5:
                checked = time.monotonic()
                if find_kernel() != kernel_id:
                    return
    finally:
        kc.stop_channels()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path.home() / "ARENA_materials" / "shared_output.ipynb")
    ap.add_argument("--max-cells", type=int, default=50, help="how many recent cells to keep (default 50)")
    ap.add_argument("--oldest-first", action="store_true", help="chronological order, newest at the bottom")
    args = ap.parse_args()
    repo = repo_of(args.out)
    if repo:
        keep_out_of_git(args.out, *repo)
        make_read_only(args.out, repo[0])
    mirror = Mirror(args.out, args.max_cells, newest_first=not args.oldest_first)
    print(f"mirroring into {args.out}", flush=True)
    while True:
        kernel_id = find_kernel()
        if kernel_id:
            print(f"following kernel {kernel_id[:8]}", flush=True)
            follow(kernel_id, mirror)
        else:
            time.sleep(3)


if __name__ == "__main__":
    main()
