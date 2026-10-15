"""Resumable API and activation caches for the persona-vector exercises.

Successful responses are cached individually. Change ARENA_CACHE_RUN to draw
a separate reproducible set of samples, or use force_refresh=True explicitly.
Change ARENA_ACTIVATION_CACHE_TAG after modifying model weights or pooling math.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import tempfile
import time
import types

CACHE_SCHEMA = 1


def _digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def cache_run_name(value=None):
    return str(value if value is not None else os.getenv("ARENA_CACHE_RUN", "default"))


class DiskCache:
    def __init__(self, directory):
        self.directory = Path(directory)

    def _path(self, kind, key, extension):
        return self.directory / kind / key[:2] / (key + extension)

    def _atomic_write(self, path, write):
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temp = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as f:
                temp = Path(f.name)
                write(f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp, path)
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)

    def read_text(self, key):
        try:
            record = json.loads(self._path("api", key, ".json").read_text())
            text = record["text"]
            if record["schema"] == CACHE_SCHEMA and record["key"] == key and isinstance(text, str) and text.strip():
                return text
        except (OSError, ValueError, TypeError, KeyError):
            pass
        return None

    def write_text(self, key, text):
        if not isinstance(text, str) or not text.strip():
            return
        record = {"schema": CACHE_SCHEMA, "key": key, "text": text}
        self._atomic_write(self._path("api", key, ".json"), lambda f: f.write(json.dumps(record, ensure_ascii=False).encode()))

    def read_tensor(self, key, hidden_size, *, kind="activations"):
        import torch
        try:
            record = torch.load(self._path(kind, key, ".pt"), map_location="cpu", weights_only=True)
            value = record["value"]
            if (record["schema"] == CACHE_SCHEMA and record["key"] == key and isinstance(value, torch.Tensor)
                    and tuple(value.shape) == (hidden_size,) and value.is_floating_point() and torch.isfinite(value).all()):
                return value
        except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, RuntimeError, EOFError, pickle.UnpicklingError):
            pass
        return None

    def write_tensor(self, key, value, *, kind="activations"):
        import torch
        value = value.detach().to(device="cpu", dtype=torch.float32)
        if value.ndim != 1 or not torch.isfinite(value).all():
            raise ValueError("Refusing to cache an invalid activation mean")
        record = {"schema": CACHE_SCHEMA, "key": key, "value": value}
        self._atomic_write(self._path(kind, key, ".pt"), lambda f: torch.save(record, f))


def api_request_key(endpoint, model, messages, max_tokens, temperature, cache_run=None, sample_index=0):
    return _digest({"schema": CACHE_SCHEMA, "kind": "chat-completion", "endpoint": str(endpoint).rstrip("/"),
                    "model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                    "run": cache_run_name(cache_run), "sample_index": sample_index})


def _activation_identity(model, tokenizer, layer, cache_run=None):
    config = model.config.to_dict()
    # Ignore provenance fields that change when the same model is reloaded.
    config.pop("_name_or_path", None)
    config.pop("transformers_version", None)
    return {"config": config,
                    "model": getattr(model.config, "_name_or_path", ""),
                    "revision": getattr(model.config, "_commit_hash", None), "dtype": str(model.dtype),
                    "tokenizer": str(getattr(tokenizer, "name_or_path", "")),
                    "tokenizer_revision": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
                    "layer": int(layer), "run": cache_run_name(cache_run),
                    "model_tag": os.getenv("ARENA_ACTIVATION_CACHE_TAG", "pretrained-v1")}


def activation_request_key(model, tokenizer, prompt, response_start, layer, cache_run=None):
    return _digest({"schema": CACHE_SCHEMA, "kind": "response-mean-float32-v1",
                    **_activation_identity(model, tokenizer, layer, cache_run),
                    "prompt": prompt, "response_start": response_start})


def formatter_fingerprint(formatter):
    """Fingerprint formatting logic without unstable notebook filenames/line numbers."""
    def describe(code):
        return {"bytecode": code.co_code.hex(), "names": code.co_names,
                "constants": [describe(x) if isinstance(x, types.CodeType) else x for x in code.co_consts]}
    functions = {formatter.__name__: describe(formatter.__code__)}
    for name in formatter.__code__.co_names:
        helper = formatter.__globals__.get(name)
        if isinstance(helper, types.FunctionType) and helper.__module__ == formatter.__module__:
            functions[name] = describe(helper.__code__)
    return _digest(functions)


def persona_request_key(model, tokenizer, system_prompt, questions, responses, layer,
                        formatter_id, cache_run=None):
    return _digest({"schema": CACHE_SCHEMA, "kind": "persona-mean-v1",
                    **_activation_identity(model, tokenizer, layer, cache_run),
                    "system_prompt": system_prompt, "questions": list(questions),
                    "responses": list(responses), "formatter": formatter_id})


def _retry_delay(error, attempt):
    headers = getattr(getattr(error, "response", None), "headers", {}) or {}
    hint = headers.get("retry-after") or headers.get("Retry-After")
    if hint:
        try:
            delay = float(hint)
        except (ValueError, TypeError):
            try:
                delay = (parsedate_to_datetime(hint) - datetime.now(timezone.utc)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                delay = None
        if delay is not None and math.isfinite(delay):
            return max(0.0, delay)
    return 2 ** attempt + random.uniform(0.0, 0.25)


def cached_parallel_chat(client, messages_list, *, model, max_tokens=128, temperature=0.7,
                         max_workers=20, cache_dir, force_refresh=False, cache_run=None, progress=None):
    """Preserve input order, save every success immediately, retry transient errors.

    Duplicate prompts within a batch get separate sample indices so an experiment
    asking for repeated samples still receives independent draws on the first run.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    cache = DiskCache(cache_dir)
    endpoint = str(getattr(client, "base_url", "https://openrouter.ai/api/v1"))
    run = cache_run_name(cache_run)
    results = [""] * len(messages_list)
    pending = []
    occurrences = Counter()
    for i, messages in enumerate(messages_list):
        base = api_request_key(endpoint, model, messages, max_tokens, temperature, run)
        sample_index = occurrences[base]
        occurrences[base] += 1
        key = api_request_key(endpoint, model, messages, max_tokens, temperature, run, sample_index)
        value = None if force_refresh else cache.read_text(key)
        if value is None:
            pending.append((i, key, messages))
        else:
            results[i] = value
    print(f"API cache: {len(messages_list)-len(pending)}/{len(messages_list)} reused; {len(pending)} new requests")
    if not pending:
        return results

    # Own the retry policy rather than multiplying it by the SDK's default retries.
    request_client = client.with_options(max_retries=0, timeout=120) if hasattr(client, "with_options") else client

    def call(item):
        i, key, messages = item
        for attempt in range(3):
            try:
                response = request_client.chat.completions.create(model=model, messages=messages,
                                                                  max_tokens=max_tokens, temperature=temperature)
                text = response.choices[0].message.content
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Empty completion")
            except Exception as error:
                status = getattr(error, "status_code", None)
                headers = getattr(getattr(error, "response", None), "headers", {}) or {}
                retryable = status is None or status in (408, 409, 429, 500, 502, 503, 504) or (
                    status == 402 and bool(headers.get("retry-after") or headers.get("Retry-After")))
                if not retryable or attempt == 2:
                    print(f"API request {i+1} failed ({'HTTP '+str(status) if status else type(error).__name__}); not cached")
                    return i, ""
                delay = _retry_delay(error, attempt)
                if delay > 60:
                    print(f"API request {i+1} needs a longer retry wait; left uncached for the next run")
                    return i, ""
                time.sleep(delay)
                continue
            # Cache write errors must not trigger a second, chargeable API request.
            cache.write_text(key, text)
            return i, text

    with ThreadPoolExecutor(max_workers=min(max_workers, len(pending))) as executor:
        futures = [executor.submit(call, item) for item in pending]
        iterator = as_completed(futures)
        if progress is not None:
            iterator = progress(iterator, total=len(pending), desc="API calls (uncached)")
        for future in iterator:
            i, value = future.result()
            results[i] = value
    failed = sum(not x.strip() for x in results)
    if failed:
        print(f"{failed} empty/failed responses remain; rerunning retries only those requests")
    return results


def seed_persona_response_cache(client, personas, questions, responses, *, cache_dir,
                                model, max_tokens=256, temperature=0.7, cache_run=None):
    """Checkpoint an already completed batch using the settings that produced it."""
    cache = DiskCache(cache_dir)
    count = 0
    occurrences = Counter()
    for name, system_prompt in personas.items():
        for question_idx, question in enumerate(questions):
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
            base = api_request_key(str(client.base_url), model, messages, max_tokens, temperature, cache_run)
            sample_index = occurrences[base]
            occurrences[base] += 1
            text = responses.get((name, question_idx))
            if not isinstance(text, str) or not text.strip():
                continue
            key = api_request_key(str(client.base_url), model, messages, max_tokens, temperature, cache_run, sample_index)
            cache.write_text(key, text)
            count += 1
    return count


def forward_to_hidden_state(model, tokens, layer: int) -> Tensor:
    """Get one HuggingFace hidden-state index without computing vocabulary logits.

    For an intermediate state, stop after block layer-1. Earlier blocks are
    unaffected by later blocks, so this is the same activation as a full pass.
    The last hidden-state entry is special: it includes the final normalization.
    """
    backbone = model.model
    if hasattr(backbone, "language_model"):
        backbone = backbone.language_model
    blocks = backbone.layers
    n_layers = len(blocks)
    if not -(n_layers + 1) <= layer <= n_layers:
        raise IndexError(f"hidden_states index {layer} is invalid for {n_layers} blocks")
    layer %= n_layers + 1

    kwargs = dict(use_cache=False, output_hidden_states=False, output_attentions=False, return_dict=True)
    if layer == n_layers:
        # HF's last entry is the post-final-norm state, not the raw last block.
        return backbone(**tokens, **kwargs).last_hidden_state

    class _ReachedRequestedLayer(Exception):
        pass

    # An early return must not skip an ancestor's weight-offload cleanup. The
    # usual device_map='auto' hooks offload individual blocks, whose Accelerate
    # post_forward has already finished before our PyTorch output hook runs.
    # If a root hook itself offloads, safely complete the backbone instead.
    def offloads(hook):
        return bool(getattr(hook, "offload", False)) or any(
            offloads(h) for h in getattr(hook, "hooks", ())
        )

    stop_early = not any(offloads(getattr(m, "_hf_hook", None)) for m in (model, backbone))
    captured = {}

    def capture_output(module, args, output):
        captured["hidden"] = output[0] if isinstance(output, tuple) else output
        if stop_early:
            raise _ReachedRequestedLayer()

    def capture_input(module, args, kwargs):
        # hidden_states[0] is the state entering the first transformer block.
        captured["hidden"] = args[0] if args else kwargs["hidden_states"]
        if stop_early:
            raise _ReachedRequestedLayer()

    handle = (
        blocks[0].register_forward_pre_hook(capture_input, with_kwargs=True)
        if layer == 0
        else blocks[layer - 1].register_forward_hook(capture_output)
    )
    try:
        try:
            backbone(**tokens, **kwargs)
        except _ReachedRequestedLayer:
            pass
    finally:
        handle.remove()
    return captured["hidden"]
