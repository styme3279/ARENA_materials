"""Helpers for the *_c.py versions of ARENA notebooks: GPU-only model loading, a disk cache for slow calls, FAST mode.

Cache rules (so a cached result is never stale):
- local generation is keyed on the prompts + generation args, which adapter is active, the code of the generation
  function, and the code + attributes of every forward hook on the model (e.g. your SteeringHook). Change any of
  them and it regenerates. Hooks it can't fingerprint (plain closures) switch caching off for that call.
- API calls are keyed on model + messages + sampling args; empty/failed responses are never stored.
- set `c.CACHE = False` to always regenerate, or delete the `.c_cache/` folder next to this file.
"""

import copy
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


def load_shared_loras(base_id: str, lora_ids: dict[str, str], dtype=t.bfloat16, need_gb: float = 30.0):
    """Load the base model once on GPU 0 and attach each LoRA to its own module tree over the *same* weights.

    Two separately loaded bf16 copies of a 14B model (59 GB) don't fit on a 46 GB A40, and `device_map="auto"` then
    silently offloads layers to CPU. Here the copies share parameters, so the cost is one model + the adapters,
    and every LoRA keeps the usual adapter name "default".

    Returns (bases, loras, tokenizer): dicts keyed like `lora_ids`; bases[k] is the module tree under loras[k].
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    free, _ = t.cuda.mem_get_info()
    if free / GB < need_gb:
        raise RuntimeError(
            f"only {free / GB:.1f} GiB free on the GPU, need ~{need_gb:.0f}. Another kernel probably still holds a "
            "model: check `nvidia-smi` and restart/close it."
        )
    base = AutoModelForCausalLM.from_pretrained(base_id, dtype=dtype, device_map={"": 0})
    shared = {id(p): p for p in base.parameters()}
    trees = [base] + [copy.deepcopy(base, memo=dict(shared)) for _ in range(len(lora_ids) - 1)]  # before any LoRA
    bases, loras = {}, {}
    for tree, (name, repo) in zip(trees, lora_ids.items()):
        bases[name] = tree
        loras[name] = PeftModel.from_pretrained(tree, repo)
        loras[name]._c_tag = repo
    tok = AutoTokenizer.from_pretrained(base_id)
    tok.pad_token = tok.eos_token
    gpu_report(*loras.values(), label="after loading")
    return bases, loras, tok


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
    tag = getattr(model, "_c_tag", None)
    if tag is None:
        return None
    for m in model.modules():
        if hasattr(m, "lora_A") and hasattr(m, "disable_adapters"):
            return [tag, not m.disable_adapters, list(m.active_adapters)]
    return [tag, False, []]


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
            why = ("model not loaded via c.load_shared_loras" if ad is None else
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
