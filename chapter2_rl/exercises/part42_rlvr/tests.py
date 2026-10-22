# RLVR tests for [2.4.2].
#
# Every test compares the student's function against the reference implementation in `solutions.py`, on synthetic,
# seeded inputs: no model, no network, CPU only, a few seconds in total. The inputs are built so that the answer never
# hinges on luck (e.g. advantages are never all zero, every clipping regime is exercised), and a "Looks like you did X"
# hint only fires when that mistake provably gives a different value from the correct one.
import contextlib
import io
import random
import re
import sys
from numbers import Real
from pathlib import Path

import torch as t

exercises_dir = Path(__file__).parent.parent
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))


def _solutions():
    """The reference solutions. Imported lazily: solutions.py imports this module at the top, so importing it here at
    module level would be circular. Its import-time prints (e.g. Liger kernels) are silenced."""
    with contextlib.redirect_stdout(io.StringIO()):
        import part42_rlvr.solutions as solutions
    return solutions


def _close(x, y, rtol=1e-5, atol=1e-6) -> bool:
    return t.allclose(t.as_tensor(x, dtype=t.float64), t.as_tensor(y, dtype=t.float64), rtol=rtol, atol=atol)


def _distinct_hint(got, expected, mistakes) -> str:
    """The first hint whose mistake value matches `got` and is clearly different from `expected`."""
    for value, hint in mistakes:
        if not _close(value, expected, rtol=1e-3, atol=1e-4) and _close(got, value):
            return hint
    return ""


# ======================================================================================================================
# Task generator
# ======================================================================================================================


def test_mult_byhand(mult_byhand):
    solutions = _solutions()
    for difficulty, (name, (a_lo, a_hi), (b_lo, b_hi)) in enumerate(solutions.DIFFICULTY_LADDER):
        for seed in range(25):
            out = mult_byhand(random.Random(seed), difficulty)
            assert isinstance(out, tuple) and len(out) == 2, "Return a tuple (prompt, answer)."
            prompt, answer = out
            assert isinstance(prompt, str), f"The prompt should be a str, got {type(prompt).__name__}."
            assert type(answer) is int, f"The answer should be a Python int, got {type(answer).__name__}."
            m = re.search(r"(\d+)\s*\*\s*(\d+)", prompt)
            assert m, f"Prompt should state the product as 'a * b', got {prompt!r}."
            a, b = int(m.group(1)), int(m.group(2))
            assert a_lo <= a <= a_hi and b_lo <= b <= b_hi, (
                f"Rung {difficulty} ({name}) should have a in [{a_lo}, {a_hi}] and b in [{b_lo}, {b_hi}], got {a} * {b}."
            )
            assert answer == a * b, f"The answer should equal a * b = {a * b}, got {answer}."
            assert "\\boxed{" in prompt, "Prompt should ask for the answer in \\boxed{...}."
            assert prompt.endswith("\n"), "Prompt should end with a newline, so the base model starts its answer on a new line."
            expected = solutions.mult_byhand(random.Random(seed), difficulty)
            assert (prompt, answer) == expected, (
                f"mult_byhand(random.Random({seed}), {difficulty}) = {(prompt, answer)!r}, expected {expected!r}. Draw "
                "a then b with `rng.randint` (inclusive on both ends), and use the prompt format given above."
            )
    # both ends of each range are reachable (e.g. `rng.randrange(lo, hi)` would never draw `hi`). Checked on the rungs
    # with at most 900 values per operand, where 20k draws reach every value with overwhelming probability
    for difficulty in (0, 2):
        _, (a_lo, a_hi), (b_lo, b_hi) = solutions.DIFFICULTY_LADDER[difficulty]
        rng = random.Random(0)
        ops = [re.search(r"(\d+)\s*\*\s*(\d+)", mult_byhand(rng, difficulty)[0]).groups() for _ in range(20_000)]
        a_seen, b_seen = {int(a) for a, _ in ops}, {int(b) for _, b in ops}
        assert {a_lo, a_hi} <= a_seen and {b_lo, b_hi} <= b_seen, (
            f"On rung {difficulty}, a should range over [{a_lo}, {a_hi}] and b over [{b_lo}, {b_hi}] inclusive, but "
            f"20,000 draws only gave a in [{min(a_seen)}, {max(a_seen)}], b in [{min(b_seen)}, {max(b_seen)}]."
        )
    # the default difficulty is the first rung, and the task varies with the rng
    assert mult_byhand(random.Random(0)) == mult_byhand(random.Random(0), 0), "The default difficulty should be 0."
    assert len({mult_byhand(random.Random(s))[1] for s in range(20)}) > 10, "Task should vary with rng."
    print("All tests in `test_mult_byhand` passed!")


# ======================================================================================================================
# Answer extraction & reward
# ======================================================================================================================

# Hand-picked cases, each with what it checks. Only behaviour the contract pins down is tested: we deliberately leave
# out cases where a reasonable implementation could go either way (commas "15,804", "+5", "- 5", a unicode minus,
# non-ASCII digits, "1_000", newlines/tabs inside the box, a space in "\boxed {5}", doubled braces "\boxed{{5}}",
# nested boxes, and a later *non-integer* or unclosed box after an integer one).
_BOXED_CASES = [
    ("the answer is \\boxed{15804}.", "a plain boxed integer"),
    ("\\boxed{42}", "a plain boxed integer"),
    ("So $\\boxed{42}$.", "a boxed integer inside $...$"),
    ("\\[ \\boxed{816} \\]", "a boxed integer inside \\[...\\]"),
    ("\\boxed{-42}", "negative integers"),
    ("work... \\boxed{-0}", "negative integers"),
    ("\\boxed{ 42 }", "spaces around the integer inside the box"),
    ("\\boxed{   -7}", "spaces around the integer inside the box"),
    ("\\boxed{-7   }", "spaces around the integer inside the box"),
    ("\\boxed{007}", "leading zeros (the integer is still 7)"),
    ("\\boxed{-0012}", "leading zeros (the integer is still -12)"),
    ("\\boxed{0}", "zero"),
    ("\\boxed{" + "9" * 60 + "}", "huge integers (use int(), not float(), which loses precision)"),
    ("\\boxed{-" + "1234567890" * 5 + "}", "huge integers (use int(), not float(), which loses precision)"),
    ("first \\boxed{100}, then \\boxed{200}", "several boxes: take the last"),
    ("\\boxed{1}\\boxed{2}\\boxed{3}", "several adjacent boxes: take the last"),
    ("\\boxed{-5} or maybe\n\nno, it's \\boxed{ 6 }", "several boxes on different lines: take the last"),
    ("\\boxed{12}} trailing brace", "an extra closing brace after the box"),
    ("x" * 5000 + " \\boxed{31} " + "y" * 5000, "a boxed integer inside a long completion"),
    ("\\boxed{1 5 8 0 4}", "digit-spaced content is not an integer"),
    ("\\boxed{hello}", "non-integer content"),
    ("\\boxed{N}", "non-integer content (the placeholder from the prompt)"),
    ("\\boxed{}", "an empty box"),
    ("\\boxed{ }", "an empty box"),
    ("\\boxed{-}", "a lone minus sign"),
    ("\\boxed{--5}", "a double minus sign"),
    ("\\boxed{4-2}", "an expression is not an integer"),
    ("\\boxed{4.5}", "decimals are not integers"),
    ("\\boxed{4.0}", "decimals are not integers"),
    ("\\boxed{1e3}", "scientific notation is not an integer"),
    ("\\boxed{42abc}", "digits followed by junk"),
    ("\\boxed{abc42}", "junk followed by digits"),
    ("\\boxed{0x1F}", "hex is not a plain integer"),
    ("\\boxed{1/2}", "fractions are not integers"),
    ("\\boxed{\\frac{1}{2}}", "nested braces / LaTeX fractions are not integers"),
    ("\\boxed{x = 5}", "an equation is not an integer"),
    ("no box here, just 4290", "only boxed answers count"),
    ("{4290}", "only boxed answers count"),
    ("\\fbox{4290}", "only \\boxed counts"),
    ("4290", "only boxed answers count"),
    ("", "the empty string"),
    ("\\boxed{42", "a box with no closing brace"),
    ("the answer is \\boxed{4290 and then I ran out of tokens", "a box with no closing brace"),
    ("the answer is \\boxed{", "a box with no closing brace"),
]

_FILLER = ["", " ", "So the answer is ", "Step 1: 12 * 3 = 36. ", "\n\n", "Hence, ", " 4290 ", "$", "(", ").", "-", "7"]
_NON_INT = ["", " ", "abc", "4.5", "1e3", "x = 5", "\\frac{1}{2}", "1 2", "--3", "N", "3-1"]


def _random_completion(rng: random.Random) -> str:
    """Random text made of filler and well-formed boxes. Either the last box holds an integer or no box does, so the
    answer never depends on the contract's grey areas (see `_BOXED_CASES`)."""
    n_boxes = rng.randint(0, 4)
    any_int = rng.random() < 0.7
    parts = [rng.choice(_FILLER)]
    for i in range(n_boxes):
        if any_int and (i == n_boxes - 1 or rng.random() < 0.5):
            digits = "0" * rng.choice([0, 0, 0, 1, 3]) + str(rng.randint(0, 10 ** rng.randint(1, 30)))
            content = " " * rng.randint(0, 2) + rng.choice(["", "", "-"]) + digits + " " * rng.randint(0, 2)
        else:
            content = rng.choice(_NON_INT)
        parts += ["\\boxed{" + content + "}", rng.choice(_FILLER)]
    return "".join(parts)


def test_extract_boxed_int(extract_boxed_int):
    solutions = _solutions()
    rng = random.Random(0)
    cases = _BOXED_CASES + [(_random_completion(rng), "random completions") for _ in range(300)]
    for text, what in cases:
        expected = solutions.extract_boxed_int(text)
        got = extract_boxed_int(text)
        shown = text if len(text) < 200 else text[:80] + " ... " + text[-80:]
        if expected is None:
            assert got is None, f"extract_boxed_int({shown!r}) = {got!r}, expected None. (Checking: {what}.)"
        else:
            assert type(got) is int, (
                f"extract_boxed_int({shown!r}) should return a Python int, got {got!r} ({type(got).__name__})."
            )
            assert got == expected, f"extract_boxed_int({shown!r}) = {got!r}, expected {expected!r}. (Checking: {what}.)"
    print("All tests in `test_extract_boxed_int` passed!")


def test_int_reward(int_reward):
    solutions = _solutions()
    groups = []
    for answer in [4290, -7, 0, 7, 10**40 + 1]:
        a = str(answer)
        groups.append((
            [
                f"work... \\boxed{{{a}}}",                              # correct -> 1.0
                f"work... \\boxed{{{answer + 1}}}",                     # boxed but wrong -> 0.1
                f"the answer is {a}",                                   # correct number but not boxed -> 0.0
                "no idea",                                              # nothing -> 0.0
                f"\\boxed{{ {a} }}",                                    # whitespace inside the box -> 1.0
                f"\\boxed{{{answer + 1}}}, no wait: \\boxed{{{a}}}",    # the *last* box counts -> 1.0
                f"\\boxed{{{a}}}, no wait: \\boxed{{{answer - 1}}}",    # the *last* box counts -> 0.1
                f"\\boxed{{{a}.5}}",                                    # not an integer -> 0.0
                f"\\boxed{{{a}",                                        # unclosed -> 0.0
                f"{a} \\boxed{{x}} {a}",                                # answer everywhere but in the box -> 0.0
                f"\\boxed{{{-answer if answer else 1}}}",               # wrong sign -> 0.1
            ],
            answer,
        ))
    groups.append((["\\boxed{0004290}", "\\boxed{04290}"], 4290))    # compare as integers, not strings -> 1.0
    groups.append((["\\boxed{4290}"], 4290))                          # G = 1
    groups.append((["\\boxed{4290}", "\\boxed{0}", "\\boxed{-0}"], 4291))  # same text, new answer; a wrong boxed 0 -> 0.1
    groups.append((["\\boxed{4290}", "\\boxed{0}"], 0))
    rng = random.Random(0)
    for _ in range(100):                                              # random groups, answers drawn from the boxes
        group = [_random_completion(rng) for _ in range(rng.randint(1, 9))]
        boxed = [solutions.extract_boxed_int(c) for c in group]
        answer = rng.choice([b for b in boxed if b is not None] or [rng.randint(-50, 50)])
        groups.append((group, answer))

    for completions, answer in groups:
        original = list(completions)
        out = int_reward(completions, answer)
        assert completions == original, "int_reward should not modify the list of completions."
        assert isinstance(out, t.Tensor), f"int_reward should return a tensor, got {type(out).__name__}"
        G = len(completions)
        assert out.shape == (G,), f"int_reward should return one reward per completion, shape ({G},), got {tuple(out.shape)}"
        assert out.is_floating_point(), f"int_reward should return a float tensor (e.g. t.tensor(list_of_floats)), got {out.dtype}"
        expected = solutions.int_reward(completions, answer)
        for i, text in enumerate(completions):
            got, exp = out[i].item(), expected[i].item()
            if abs(got - exp) < 1e-6:
                continue
            hint = ""
            if exp == 0.0 and got == 1.0:
                hint = " Only a well-formed boxed integer counts: use `extract_boxed_int`, don't search the raw text."
            elif exp == 0.0 and abs(got - 0.1) < 1e-6:
                hint = " 0.1 is only for a well-formed boxed integer that's wrong; `extract_boxed_int` returns None otherwise."
            elif abs(exp - 0.1) < 1e-6 and got == 0.0:
                hint = " A well-formed boxed integer that's wrong should get 0.1, not 0.0."
            elif exp == 1.0 and abs(got - 0.1) < 1e-6:
                hint = " Compare the extracted int with `answer` as integers (e.g. '\\boxed{007}' is 7)."
            raise AssertionError(
                f"int_reward[{i}] = {got}, expected {exp} for completion {text!r} and answer {answer}.{hint}"
            )
    print("All tests in `test_int_reward` passed!")


# ======================================================================================================================
# Curriculum
# ======================================================================================================================


def test_curriculum_update(Curriculum):
    solutions = _solutions()

    def make(cls, initial, lr=2.0, target=0.3, d_min=0.0, d_max=5.0):
        return cls(task=None, reward_fn=None, initial=initial, lr=lr, target=target, d_min=d_min, d_max=d_max)

    def check_step(c, ref, solve, desc):
        d, lr, target, d_min, d_max = ref.difficulty, ref.lr, ref.target, ref.d_min, ref.d_max
        state = (c.lr, c.target, c.d_min, c.d_max, c.rng.getstate(), c.np_rng.bit_generator.state)
        c.update(solve)
        ref.update(solve)
        assert (c.lr, c.target, c.d_min, c.d_max, c.rng.getstate(), c.np_rng.bit_generator.state) == state, (
            "`update` should only change `self.difficulty` (not lr, target, d_min, d_max, or the rngs)."
        )
        got, expected = c.difficulty, ref.difficulty
        assert isinstance(got, Real) and not isinstance(got, bool), (
            f"`self.difficulty` should stay a number (float), got {type(got).__name__}. Did `update` return the new "
            "value instead of setting `self.difficulty`?"
        )
        if abs(got - expected) <= 1e-6 * max(1.0, abs(expected)):
            return

        def clip(x):
            return min(max(x, d_min), d_max)

        hint = _distinct_hint(got, expected, [
            (clip(d - lr * (solve - target)),
             "D moved the wrong way: a batch solved more often than the target should make D *harder* (larger)."),
            (clip(d + lr * (solve - 0.5)), "Use `self.target`, not a hard-coded 0.5."),
            (clip(d + (solve - target)), "Scale the step by `self.lr`."),
            (d + lr * (solve - target), "Clamp D to [self.d_min, self.d_max] after the step."),
            (min(d + lr * (solve - target), d_max), "Clamp D from below too, at `self.d_min`."),
            (max(d + lr * (solve - target), d_min), "Clamp D from above too, at `self.d_max`."),
            (clip(d) + lr * (solve - target), "Clamp *after* taking the step, not before."),
        ])
        raise AssertionError(
            f"{desc}: from D={d}, update({solve}) with lr={lr}, target={target}, range [{d_min}, {d_max}] should give "
            f"D -> {expected}, got {got}.{(' ' + hint) if hint else ''}"
        )

    # hand-picked: up, down, at the target, clamped above and below
    for initial, solve in [(2.0, 0.8), (2.0, 0.1), (2.0, 0.3), (4.5, 1.0), (0.2, 0.0), (5.0, 0.9), (0.0, 0.2)]:
        check_step(make(Curriculum, initial), make(solutions.Curriculum, initial), solve, "Single update")

    # D saturates at d_max, then the model falls behind: D should come straight back down from d_max (the clamp is part
    # of the state, not just of what's reported)
    c, ref = make(Curriculum, 4.0), make(solutions.Curriculum, 4.0)
    for solve in [1.0, 1.0, 1.0, 0.0, 0.0, 0.25, 0.3, 1.0, 0.0, 0.0, 0.0, 0.0, 0.1]:
        check_step(c, ref, solve, "Saturating sequence")

    # random configs and sequences of solve rates, including exact hits on the target and the clamp bounds
    rng = random.Random(0)
    for trial in range(60):
        d_min = rng.choice([0.0, 0.0, -1.5, rng.uniform(-2, 2)])
        d_max = d_min + rng.choice([0.0, rng.uniform(0.5, 8.0), 7.0])
        target = rng.choice([0.5, rng.uniform(0.05, 0.95)])
        lr = rng.choice([0.0, 1.0, rng.uniform(0.1, 6.0)])
        initial = rng.choice([d_min, d_max, rng.uniform(d_min, d_max)])
        c = make(Curriculum, initial, lr, target, d_min, d_max)
        ref = make(solutions.Curriculum, initial, lr, target, d_min, d_max)
        for _ in range(30):
            G = rng.randint(1, 64)
            solve = rng.choice([0.0, 1.0, target, rng.randint(0, G) / G, rng.random()])
            check_step(c, ref, solve, f"Random sequence {trial}")

    c = make(Curriculum, 1.7, lr=0.0)
    for solve in [0.95, 0.0, 0.3]:
        c.update(solve)
    assert abs(c.difficulty - 1.7) < 1e-6, f"With lr=0 the difficulty should stay fixed, got {c.difficulty}"
    print("All tests in `test_curriculum_update` passed!")


# ======================================================================================================================
# Completion mask
# ======================================================================================================================


def _mask_battery(seed=0):
    """(gen, prefix_len, eos_id) cases: left-padded prompts that may contain EOS (Qwen's pad == eos), completions that
    stop immediately / later / never, EOS-as-padding, non-EOS padding or junk after the EOS, and several EOS."""
    g = t.Generator().manual_seed(seed)
    cases = []
    for prefix_len, gen_len, eos_id in [(4, 9, 9), (0, 7, 3), (6, 1, 0), (5, 2, 2), (3, 12, 50), (7, 6, 1), (1, 15, 11)]:
        rows = []
        for kind in range(14):
            prompt = t.randint(0, 60, (prefix_len,), generator=g)
            n_pad = int(t.randint(0, prefix_len + 1, (), generator=g))
            prompt[:n_pad] = eos_id if kind % 2 == 0 else 59             # left padding: EOS-as-pad half the time
            comp = t.randint(0, 60, (gen_len,), generator=g)
            comp[comp == eos_id] = (eos_id + 1) % 60                     # start with no EOS at all
            stop = int(t.randint(0, gen_len, (), generator=g))
            if kind in (1, 2, 3):                                      # EOS as the very first generated token
                comp[0] = eos_id
                if kind == 1:
                    comp[1:] = eos_id
            elif kind in (4, 5):                                         # EOS at the last position
                comp[-1] = eos_id
            elif kind in (6, 7, 8):                                      # EOS then EOS-as-padding
                comp[stop:] = eos_id
            elif kind in (9, 10):                                        # EOS then more EOS among other tokens
                comp[stop] = eos_id
                comp[stop + 1:][t.rand(gen_len - stop - 1, generator=g) < 0.4] = eos_id
            elif kind == 11:                                             # EOS then a different pad token
                comp[stop] = eos_id
                comp[stop + 1:] = (eos_id + 7) % 60
            rows.append(t.cat([prompt, comp]))                           # kinds 0, 12, 13: never stops
        cases.append((t.stack(rows), prefix_len, eos_id))
    return cases


def test_completion_mask(completion_mask):
    solutions = _solutions()
    P, EOS = 0, 9              # prompt is 4 tokens; finished rows are right-padded with EOS, as for Qwen (pad == eos)
    gen = t.tensor([
        [EOS, 5, 6, 7, 1, 2, 9, 9, 9],   # left-padded prompt (pad == EOS), 2 tokens, EOS, then EOS-as-padding
        [4, 5, 6, 7, 1, 9, 9, 9, 9],     # stops immediately after 1 token
        [4, 5, 6, 7, 1, 2, 3, 2, 1],     # never stops (truncated at the budget)
        [4, 5, 6, 7, 9, 9, 9, 9, 9],     # EOS as the very first generated token
        [P, P, 6, 7, 1, 9, 3, 9, 3],     # junk after the first EOS
    ])
    expected = t.tensor([               # completion columns only: gen[:, 4:]
        [1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
    ], dtype=t.float32)
    assert t.equal(solutions.completion_mask(gen, 4, EOS), expected)   # sanity check on the hand-written case
    for gen, prefix_len, eos_id in [(gen, 4, EOS)] + _mask_battery():
        gen_copy = gen.clone()
        out = completion_mask(gen, prefix_len, eos_id)
        assert t.equal(gen, gen_copy), "completion_mask should not modify `gen` in place."
        ref = solutions.completion_mask(gen, prefix_len, eos_id)
        assert isinstance(out, t.Tensor), f"completion_mask should return a tensor, got {type(out).__name__}"
        if out.shape == gen.shape and prefix_len > 0:
            raise AssertionError("completion_mask should cover only the completion, gen[:, prefix_len:], not the prompt too.")
        assert out.shape == ref.shape, (
            f"completion_mask should return shape {tuple(ref.shape)} = (batch, seq - prefix_len), got {tuple(out.shape)}"
        )
        assert out.is_floating_point(), f"completion_mask should return a float mask (use `.float()`), got {out.dtype}"
        for i in range(len(gen)):
            if t.equal(out[i], ref[i]):
                continue
            comp = gen[i, prefix_len:]
            hint = ""
            if t.equal(out[i], ref[i] * (comp != eos_id).float()):
                hint = " The first EOS is missing from the mask: is it being treated as padding (pad == eos)?"
            elif (comp == eos_id).sum() == 0 and out[i].sum() < len(comp):
                hint = " This completion has no EOS (it was truncated at the budget): every generated token counts."
            elif (gen[i, :prefix_len] == eos_id).any():
                hint = " EOS tokens in the prompt (left padding, since pad == eos) shouldn't affect the completion's mask."
            elif out[i].sum() > ref[i].sum():
                hint = " Everything after the *first* EOS should be masked out, including any later non-EOS tokens."
            raise AssertionError(
                f"prefix_len={prefix_len}, eos_id={eos_id}, row {i}: gen={gen[i].tolist()}\n"
                f"  expected {ref[i].int().tolist()}\n  got      {out[i].int().tolist()}.{hint}"
            )
    print("All tests in `test_completion_mask` passed!")


# ======================================================================================================================
# Dr. GRPO objective
# ======================================================================================================================


def _token_logprobs(g, shape, scale, perturb, vocab=40):
    """Logprobs of random tokens under log_softmax(scale * Gaussian logits) ("old"), and of the same tokens under
    logits nudged by `perturb * Gaussian` ("new": a policy a few gradient steps away)."""
    logits = t.randn(*shape, vocab, generator=g) * scale
    tokens = t.randint(vocab, (*shape, 1), generator=g)
    new_logits = logits + t.randn(*shape, vocab, generator=g) * perturb
    return logits.log_softmax(-1).gather(-1, tokens)[..., 0], new_logits.log_softmax(-1).gather(-1, tokens)[..., 0]


def _drgrpo_configs():
    """Synthetic (new, old, adv, mask, max_gen_len, clip) cases. batch, gen_len and max_gen_len are all distinct (so
    wrong-dim bugs and wrong normalisers show), advantages are non-degenerate with mixed signs, completion lengths
    vary (including 0 and full length), and masked-out positions hold junk logprobs that must not leak in."""
    g = t.Generator().manual_seed(0)
    configs = []
    for batch, gen_len, max_gen_len, clip, scale, perturb in [
        (7, 13, 19, 0.2, 1.0, 0.3),
        (5, 11, 16, 0.1, 30.0, 0.5),       # very peaked distributions: logprobs in the hundreds
        (9, 6, 10, 0.3, 3.0, 1.5),
        (4, 17, 23, 0.2, 0.1, 3.0),        # nearly uniform distributions, big policy moves
    ]:
        old, new = _token_logprobs(g, (batch, gen_len), scale, perturb)
        adv = t.randn(batch, generator=g).abs() * 1.3 + 0.05
        adv[1::2] *= -1                                                # both signs, alternating
        lengths = t.randint(1, gen_len, (batch,), generator=g)
        lengths[0], lengths[-1] = gen_len, 0                           # a full-length row and an empty row
        mask = (t.arange(gen_len)[None] < lengths[:, None]).float()
        new = t.where(mask.bool(), new, new + t.empty_like(new).uniform_(-4, 4, generator=g))
        configs.append((new, old, adv, mask, max_gen_len, clip))
    # a non-prefix (arbitrary 0/1) mask, with Dr. GRPO advantages computed from 0 / 0.1 / 1 rewards
    old, new = _token_logprobs(g, (6, 8), 2.0, 1.0)
    rewards = t.tensor([[1.0, 0.1, 0.0], [0.0, 0.0, 1.0]])
    adv = (rewards - rewards.mean(-1, keepdim=True)).flatten()
    mask = (t.rand(6, 8, generator=g) < 0.6).float()
    configs.append((new, old, adv, mask, 12, 0.2))
    return configs


def _drgrpo_mistakes(new, old, adv, mask, max_gen_len, clip):
    """(value, hint) pairs for common mistakes."""
    batch, seq_len = mask.shape
    A = adv[:, None]

    def surr(new=new, old=old, A=A, mask=mask):
        ratio = (new - old).exp()
        return t.minimum(ratio * A, ratio.clamp(1 - clip, 1 + clip) * A) * mask

    s = surr()
    ratio = (new - old).exp()
    lengths = mask.sum(1).clamp(min=1)
    return [
        (s.sum() / max_gen_len, "Wrong sign: return the *loss* to minimise, i.e. the negative of the objective."),
        (-s.sum() / (batch * max_gen_len),
         "You also divided by `batch`. Return the *sum* over completions, divided only by `max_gen_len`: the trainer "
         "computes the loss a micro-batch at a time and divides by the number of completions in the whole buffer."),
        (-s.sum() / mask.sum(),
         "You normalised by `mask.sum()` (the actual token count). That's GRPO's length bias; Dr. GRPO divides "
         "by the constant `max_gen_len`."),
        (-(s.sum(1) / lengths).sum(),
         "You divided each sequence by its own length. That's GRPO's length bias; Dr. GRPO divides every sequence "
         "by the same constant, `max_gen_len`."),
        (-(s.sum(1) / lengths).mean(),
         "You took a per-sequence mean then averaged over the batch. That's GRPO's length bias; Dr. GRPO sums "
         "every token and divides by the constant `max_gen_len`."),
        (-s.sum() / seq_len,
         "You divided by the tensors' sequence length. Use `max_gen_len`, the generation budget, which can be "
         "longer than the tensors you're given."),
        (-s.mean(), "You took the mean over all tokens (i.e. divided by `mask.numel()`). Sum, then divide by "
         "`max_gen_len` only."),
        (-s.sum(), "You didn't normalise. Divide the summed loss by `max_gen_len`."),
        (-surr(mask=t.ones_like(mask)).sum() / max_gen_len,
         "Looks like the mask wasn't applied: padding tokens after the end of each completion are leaking in."),
        (-surr(A=(adv / adv.std())[:, None]).sum() / max_gen_len,
         "Looks like you divided the advantages by their std. Dr. GRPO uses the advantages as given (no std "
         "normalisation)."),
        (-surr(A=((adv - adv.mean()) / adv.std())[:, None]).sum() / max_gen_len,
         "Looks like you standardised the advantages, (adv - mean) / std. Dr. GRPO uses the advantages as given."),
        (-surr(A=(adv - adv.mean())[:, None]).sum() / max_gen_len,
         "Looks like you re-centred the advantages. Use them as given: they're already group-relative."),
        (-(ratio * A * mask).sum() / max_gen_len,
         "No clipping: use t.minimum(ratio * adv, clamp(ratio, 1 - clip, 1 + clip) * adv)."),
        (-(ratio.clamp(1 - clip, 1 + clip) * A * mask).sum() / max_gen_len,
         "Only the clipped term was used. Take the elementwise *minimum* of the clipped and unclipped terms."),
        (-(t.maximum(ratio * A, ratio.clamp(1 - clip, 1 + clip) * A) * mask).sum() / max_gen_len,
         "You took the maximum of the clipped and unclipped terms; PPO's surrogate takes the *minimum* "
         "(a pessimistic bound)."),
        (-surr(new=old, old=new).sum() / max_gen_len,
         "The ratio is upside down: it should be pi_new / pi_old = exp(new_logprobs - old_logprobs)."),
        (-(t.minimum((new - old) * A, (new - old).clamp(-clip, clip) * A) * mask).sum() / max_gen_len,
         "You used the log-ratio new - old directly: the ratio is exp(new_logprobs - old_logprobs)."),
    ]


def _check_drgrpo_case(drgrpo_objective, solutions, case, desc, check_grad=True, mistakes=(), pass_clip=True):
    """Compare value, dtype, gradient w.r.t. `new_logprobs`, and input immutability against the reference."""
    new, old, adv, mask, max_gen_len, clip = case
    copies = [x.clone() for x in (new, old, adv, mask)]
    new_g = new.clone().requires_grad_()
    try:
        out = drgrpo_objective(new_g, old, adv, mask, max_gen_len, **({"clip": clip} if pass_clip else {}))
    except RuntimeError as e:
        if "size of tensor" in str(e) or "must match" in str(e):
            raise AssertionError(
                f"Shape error: {e}\n`advantages` has shape (batch,) but the logprobs are (batch, gen_len). "
                "Broadcast one advantage over each sequence with `advantages[:, None]`."
            ) from e
        raise
    assert all(t.equal(x, c) for x, c in zip((new_g.detach(), old, adv, mask), copies)), (
        "drgrpo_objective should not modify its inputs in place."
    )
    assert isinstance(out, t.Tensor) and out.dim() == 0, (
        f"drgrpo_objective should return a scalar tensor, got {type(out).__name__} of shape "
        f"{tuple(getattr(out, 'shape', ()))}. Sum over every token and divide by max_gen_len."
    )
    assert out.is_floating_point(), f"The loss should be a float tensor, got {out.dtype}."

    ref_new = new.clone().requires_grad_()
    expected = solutions.drgrpo_objective(ref_new, old, adv, mask, max_gen_len, clip=clip)
    if not _close(out.detach(), expected.detach()):
        hint = _distinct_hint(out.detach(), expected.detach(), mistakes) or (
            "Check: ratio = exp(new - old), surrogate = min(ratio * adv, clamp(ratio, 1-clip, 1+clip) * adv), "
            "masked, summed over every token, divided by max_gen_len, and negated."
        )
        raise AssertionError(f"{desc}: drgrpo_objective = {out.item():.6g}, expected {expected.item():.6g}.\n{hint}")
    if not check_grad:
        return
    assert out.requires_grad, "The loss has no gradient: don't detach `new_logprobs` (or the loss), or convert to Python numbers."
    out.backward()
    expected.backward()
    assert new_g.grad is not None, "No gradient reached `new_logprobs`."
    assert t.isfinite(new_g.grad).all(), f"{desc}: the gradient w.r.t. `new_logprobs` has NaNs / infs."
    assert (new_g.grad * (1 - mask)).abs().sum() == 0, f"{desc}: masked-out tokens should get zero gradient."
    assert t.allclose(new_g.grad, ref_new.grad, rtol=1e-4, atol=1e-7), (
        f"{desc}: the loss value is right but its gradient w.r.t. `new_logprobs` isn't (max abs error "
        f"{(new_g.grad - ref_new.grad).abs().max().item():.3g}). Are you detaching part of the computation, e.g. the "
        "ratio, or using a non-differentiable op?"
    )


@t.enable_grad()  # the gradient checks need autograd, even if the caller has it turned off
def test_drgrpo_objective(drgrpo_objective):
    solutions = _solutions()
    configs = _drgrpo_configs()

    # sanity check on the test itself: the first case hits every clipping regime, with both advantage signs
    new, old, adv, mask, _, clip = configs[0]
    ratio, live = (new - old).exp(), mask.bool()
    for sign in (1, -1):
        for regime in (ratio > 1 + clip, ratio < 1 - clip, (ratio - 1).abs() < clip):
            assert (regime & live & (sign * adv[:, None] > 0)).any(), "test bug: a clipping regime isn't exercised"

    for i, case in enumerate(configs):
        _check_drgrpo_case(drgrpo_objective, solutions, case, f"Case {i}", mistakes=_drgrpo_mistakes(*case))
    # the default clip is 0.2
    new, old, adv, mask, max_gen_len, _ = configs[0]
    _check_drgrpo_case(drgrpo_objective, solutions, (new, old, adv, mask, max_gen_len, 0.2), "Default clip",
                       pass_clip=False)
    # on-policy (old = new.detach(), as on the first learning step): the ratio is exactly 1
    case = (new, new.clone(), adv, mask, max_gen_len, 0.2)
    _check_drgrpo_case(drgrpo_objective, solutions, case, "On-policy", mistakes=_drgrpo_mistakes(*case))

    _test_drgrpo_stability(drgrpo_objective, solutions)
    print("All tests in `test_drgrpo_objective` passed!")


def _test_drgrpo_stability(drgrpo_objective, solutions):
    """Extreme but legitimate inputs. Every case here is one the reference solution handles (finite values and
    gradients); cases where the reference itself returns inf / NaN (e.g. huge junk logprobs on masked-out tokens,
    or gradients through an overflowing ratio) are deliberately not tested."""
    g = t.Generator().manual_seed(1)
    batch, gen_len, max_gen_len = 6, 9, 14
    adv = t.randn(batch, generator=g)
    adv[0], adv[1] = 1.0, -1.0
    mask = (t.arange(gen_len)[None] < t.tensor([9, 4, 7, 1, 0, 9])[:, None]).float()

    # very unlikely tokens: logprobs around -1e4 differing by O(1). exp(new) / exp(old) would be 0 / 0
    old = -1e4 + t.randn(batch, gen_len, generator=g) * 50
    new = old + t.randn(batch, gen_len, generator=g) * 0.5
    _check_drgrpo_case(drgrpo_objective, solutions, (new, old, adv, mask, max_gen_len, 0.2),
                       "Stability (logprobs ~ -1e4: compute the ratio as exp(new - old), not exp(new) / exp(old))")

    # log-ratios of +-200: exp(200) overflows float32 to inf. Where adv > 0 the min picks the finite clipped term, so
    # the loss is finite (we only check its value, since the gradient through an inf ratio is NaN even for the
    # reference). With a log-ratio of -200 the ratio underflows to 0, and value and gradient stay finite for either sign.
    log_ratio = t.where(adv[:, None] > 0, 200.0, -200.0).expand(batch, gen_len).clone()
    log_ratio[:, ::2] = -200.0
    old = -t.rand(batch, gen_len, generator=g) * 5
    _check_drgrpo_case(drgrpo_objective, solutions, (old + log_ratio, old, adv, mask, max_gen_len, 0.2),
                       "Stability (overflowing ratio with adv > 0)", check_grad=False)
    _check_drgrpo_case(drgrpo_objective, solutions, (old - 200.0, old, adv, mask, max_gen_len, 0.2),
                       "Stability (underflowing ratio)")

    # negative advantages with large but finite ratios (log-ratio up to +9): the unclipped term is selected, with a big
    # gradient. Catches log-ratio clamps / ratio caps / dual-clip variants
    lr_big = t.linspace(3.0, 9.0, batch * gen_len).view(batch, gen_len)
    _check_drgrpo_case(drgrpo_objective, solutions, (old + lr_big, old, -adv.abs() - 0.1, mask, max_gen_len, 0.2),
                       "Large ratios with negative advantages")
    # everything masked out (e.g. a micro-batch of empty completions): the loss is exactly 0, with zero gradient
    new = old + t.randn(batch, gen_len, generator=g) * 3
    _check_drgrpo_case(drgrpo_objective, solutions, (new, old, adv, t.zeros_like(mask), max_gen_len, 0.2),
                       "All tokens masked out")
    # all advantages zero (every completion in each group got the same reward): the loss is exactly 0. Normalising the
    # advantages by their std would give 0 / 0 = NaN here.
    _check_drgrpo_case(drgrpo_objective, solutions, (new, old, t.zeros_like(adv), mask, max_gen_len, 0.2),
                       "All advantages zero (are you normalising the advantages by their std?)")
