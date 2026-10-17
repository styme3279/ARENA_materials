"""Helpers for the *_c.py versions of ARENA notebooks: a GPU memory check, a disk cache for slow calls, FAST mode.

Cache rules (so a cached result is never stale):
- local generation is keyed on the prompts + generation args, which adapter is active, the code of the generation
  function, and the code + attributes of every forward hook on the model (e.g. your SteeringHook). Change any of
  them and it regenerates. Hooks it can't fingerprint (plain closures) switch caching off for that call.
- API calls are keyed on model + messages + sampling args; empty/failed responses are never stored.
- set `c.CACHE = False` to always regenerate, or delete the `.c_cache/` folder next to this file.
"""

import functools
import hashlib
import inspect
import json
import os
import threading
import types
from pathlib import Path

import torch as t

FAST = False  # True -> smaller sweeps / fewer samples (see `pick`)
CACHE = True
CACHE_DIR = Path(__file__).resolve().parent / ".c_cache"
GB = 1024**3
_stats = {"hit": 0, "miss": 0}
_uncached: dict[str, int] = {}
_announced = False


def pick(full, fast):
    """`full` normally, `fast` when FAST is on."""
    return fast if FAST else full


# ---------------------------------------------------------------- GPU


def gpu_report(*models, label: str = ""):
    """Print GPU memory use and fail loudly if any weight of `models` is off the GPU (i.e. spilled to CPU/disk)."""
    devs = sorted({str(p.device) for m in models for p in m.parameters()})
    free, total = t.cuda.mem_get_info()
    print(
        f"[gpu{' ' + label if label else ''}] weights on {devs or '-'} | this process: "
        f"{t.cuda.memory_allocated() / GB:.1f} GiB now, {t.cuda.max_memory_allocated() / GB:.1f} GiB peak | "
        f"card: {(total - free) / GB:.1f} / {total / GB:.1f} GiB used"
    )
    if models and devs != ["cuda:0"]:
        raise RuntimeError(f"some weights are not on the GPU ({devs}) -- the model spilled over")


# ---------------------------------------------------------------- fingerprints


def _sha(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


def _code_fp(code: types.CodeType) -> str:
    """Hash of what a function does (bytecode, constants, names), ignoring file name and line numbers."""
    parts = [code.co_code.hex(), repr(code.co_names), repr(code.co_varnames)]
    for k in code.co_consts:
        parts.append(_code_fp(k) if isinstance(k, types.CodeType) else repr(k))
    return _sha("|".join(parts))


def _jsonable(x):
    if isinstance(x, t.Tensor):
        return {"tensor": hashlib.sha1(x.detach().float().cpu().numpy().tobytes()).hexdigest(), "shape": list(x.shape)}
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if hasattr(x, "tolist"):  # numpy / pandas
        return _jsonable(x.tolist())
    raise TypeError(f"can't fingerprint {type(x)}")


def _class_fp(cls) -> str:
    fns = []
    for klass in cls.__mro__[:-1]:
        for name, v in sorted(vars(klass).items()):
            if isinstance(v, types.FunctionType):
                fns.append(f"{klass.__name__}.{name}:{_code_fp(v.__code__)}")
    return _sha("|".join(fns))


def _hook_state(model):
    """Fingerprint of all forward hooks on `model`; None if one can't be fingerprinted (then we don't cache)."""
    mm = t.nn.modules.module
    if mm._global_forward_hooks or mm._global_forward_pre_hooks:
        return None
    out = []
    for name, m in model.named_modules():
        for kind, hooks in (("fwd", m._forward_hooks), ("pre", m._forward_pre_hooks)):
            for h in hooks.values():
                if getattr(h, "__module__", "") == "transformers.utils.output_capturing":
                    continue  # transformers 5 adds these on the first output_hidden_states=True pass; read-only
                owner = getattr(h, "__self__", None)
                if owner is None:
                    return None
                try:
                    state = _jsonable({k: v for k, v in vars(owner).items() if k != "hook"})
                except TypeError:
                    return None
                out.append([name, kind, h.__name__, _class_fp(type(owner)), state])
    return out


def _adapter_state(model):
    """Which LoRA the model has and whether it's on: names, shapes and first values of the LoRA weights."""
    lora = [(n, p) for n, p in model.named_parameters() if "lora_" in n]
    if not lora:
        return None
    fp = _sha(json.dumps([[n, list(p.shape), p.detach().flatten()[:8].float().tolist()] for n, p in lora]))
    for m in model.modules():
        if hasattr(m, "lora_A") and hasattr(m, "disable_adapters"):
            return [fp, not m.disable_adapters, list(m.active_adapters)]
    return [fp, False, []]


# ---------------------------------------------------------------- disk cache

_lock = threading.Lock()


def _cache_io(kind: str, key: str, compute, ok=lambda v: True):
    global _announced
    path = CACHE_DIR / kind / f"{key}.json"
    if path.exists():
        with _lock:
            _stats["hit"] += 1
            if not _announced:
                _announced = True
                print(f"[c] reusing cached outputs from {CACHE_DIR} (set c.CACHE = False to resample)")
        return json.loads(path.read_text())["value"]
    value = compute()
    with _lock:
        _stats["miss"] += 1
    if ok(value):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{os.getpid()}_{threading.get_ident()}")
        tmp.write_text(json.dumps({"value": value}))
        os.replace(tmp, path)
    return value


def cached_generation(fn):
    """Decorator for local generation helpers taking `model` and `tokenizer` arguments."""
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not CACHE:
            return fn(*args, **kwargs)
        b = sig.bind(*args, **kwargs)
        b.apply_defaults()
        model = b.arguments["model"]
        ad, hooks = _adapter_state(model), _hook_state(model)
        rest = {k: v for k, v in b.arguments.items() if k not in ("model", "tokenizer")}
        try:
            rest = _jsonable(rest)
        except TypeError:
            rest = None
        if ad is None or hooks is None or rest is None:
            why = ("model without a LoRA adapter" if ad is None else
                   "a forward hook that is a plain function/closure" if hooks is None else "arguments it can't hash")
            _uncached[why] = _uncached.get(why, 0) + 1
            return fn(*args, **kwargs)
        key = _sha(json.dumps([fn.__name__, _code_fp(fn.__code__), ad, hooks, rest], sort_keys=True))
        return _cache_io("gen", key, lambda: fn(*args, **kwargs))

    return wrapper


def cached_api(fn):
    """Decorator for single API-call helpers; never stores empty or failed responses."""
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not CACHE:
            return fn(*args, **kwargs)
        b = sig.bind(*args, **kwargs)
        b.apply_defaults()
        rest = {k: v for k, v in b.arguments.items() if k != "max_retries"}
        key = _sha(json.dumps([fn.__name__, _jsonable(rest)], sort_keys=True))
        return _cache_io("api", key, lambda: fn(*args, **kwargs), ok=lambda v: isinstance(v, str) and v != "")

    return wrapper


def cached_torch(name: str, compute):
    """Cache any picklable object (e.g. downloaded checkpoint tensors) under .c_cache/torch/<name>.pt."""
    path = CACHE_DIR / "torch" / f"{name.replace('/', '__')}.pt"
    if CACHE and path.exists():
        return t.load(path, weights_only=False)
    value = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    t.save(value, path)
    return value


def cache_stats():
    print(f"[c] cache: {_stats['hit']} reused, {_stats['miss']} computed")
    for why, n in _uncached.items():
        print(f"[c]   {n} generation calls not cached: {why}")
