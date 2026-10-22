import contextlib
import functools
import gc
import io
import tempfile
import zlib

import torch as t


def _solutions():
    """
    Imports the reference solutions. We do this inside each test rather than at the top of the file, because
    solutions.py itself imports this file. Importing them changes a couple of global torch settings, which we put back.
    """
    precision, n_threads = t.get_float32_matmul_precision(), t.get_num_threads()
    import part4_rlhf.solutions as solutions

    t.set_float32_matmul_precision(precision)
    t.set_num_threads(n_threads)
    return solutions


def _fork_rng():
    """Lets a test use the global RNG (e.g. for sampling) without changing the student's RNG state."""
    return t.random.fork_rng(devices=[t.cuda.current_device()] if t.cuda.is_available() else [])


def _with_forked_rng(test):
    """Decorator: runs `test` inside `_fork_rng()`, so that its seeding doesn't reset the student's RNG."""

    @functools.wraps(test)
    def wrapper(*args, **kwargs):
        with _fork_rng():
            return test(*args, **kwargs)

    return wrapper


def _check(
    actual: t.Tensor,
    expected: t.Tensor,
    mistakes: list,
    name: str,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    drop_indistinct: bool = False,
):
    """
    Checks `actual` against `expected`. If they don't match, raises the hint of the first common mistake whose value
    matches `actual`. Every mistake value is checked to be distinct from `expected` first (or, with drop_indistinct, the
    ones which aren't are dropped), so that a hint can never fire on a correct answer.
    """
    assert isinstance(actual, t.Tensor), f"{name} should be a tensor, got {type(actual)}"
    assert actual.shape == expected.shape, f"{name} should have shape {tuple(expected.shape)}, got {tuple(actual.shape)}"
    assert actual.is_floating_point(), f"{name} should be a float tensor, got {actual.dtype}"
    actual, expected = actual.detach().to(expected.device, expected.dtype), expected.detach()
    distinct = [not t.allclose(value.detach(), expected, atol=atol, rtol=rtol) for value, _ in mistakes]
    if drop_indistinct:
        mistakes = [m for m, d in zip(mistakes, distinct) if d]
    else:
        assert all(distinct), f"Test setup error: {[h for (_, h), d in zip(mistakes, distinct) if not d]} not distinct"
    if t.allclose(actual, expected, atol=atol, rtol=rtol):
        return
    for value, hint in mistakes:
        if t.allclose(actual, value.detach(), atol=atol, rtol=rtol):
            raise Exception(hint)
    message = f"Wrong value for {name[0].lower() + name[1:]}."
    t.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=lambda m: f"{message}\n{m}")


def _sampled_logprobs(logits: t.Tensor, tokens: t.Tensor) -> t.Tensor:
    """Log-probabilities of `tokens` (shape (batch, seq)) under `logits` (shape (batch, seq, d_vocab))."""
    return logits.log_softmax(-1).gather(-1, tokens[..., None]).squeeze(-1)


def test_compute_grpo_advantages(compute_grpo_advantages):
    solutions = _solutions()
    g = t.Generator().manual_seed(0)

    # Rewards with a nonzero mean and non-unit std (so forgetting to subtract the mean or divide by the std shows up), for
    # a few group sizes (with n = 2, the biased and unbiased std differ by a factor of sqrt(2))
    for n in [7, 64, 2]:
        rewards = t.randn(n, generator=g) * 3 + 5
        rewards_copy = rewards.clone()
        advantages = compute_grpo_advantages(rewards)
        assert t.equal(rewards, rewards_copy), "compute_grpo_advantages shouldn't modify the rewards in place"
        assert isinstance(advantages, t.Tensor) and advantages.is_floating_point(), (
            f"Expected a float tensor, got {getattr(advantages, 'dtype', type(advantages))}"
        )
        centered = rewards - rewards.mean()
        _check(
            advantages,
            solutions.compute_grpo_advantages(rewards),
            [
                (centered, "Looks like you only subtracted the mean. GRPO also divides by the std of the rewards."),
                (rewards / (rewards.std() + 1e-6), "Looks like you didn't subtract the mean of the rewards."),
                (centered / (rewards.std(unbiased=False) + 1e-6), "Use the unbiased std (PyTorch's default for `.std()`)."),
                (centered / (rewards.max() - rewards.min()), "Normalize by the std of the rewards, not their range."),
            ],
            "The advantages",
        )

    # Rewards that barely vary (std comparable to eps), so the eps term matters: it should be added to the std
    rewards = t.randn(16, generator=g) * 1e-6
    centered = rewards - rewards.mean()
    _check(
        compute_grpo_advantages(rewards),
        solutions.compute_grpo_advantages(rewards),
        [
            (centered / rewards.std(), "Looks like you didn't add eps to the std: divide by (std + eps)."),
            (centered / (rewards.var() + 1e-6).sqrt(), "Add eps to the std itself, not to the variance."),
        ],
        "The advantages (for rewards with a tiny std)",
    )

    # A non-default eps should be used, not ignored
    rewards = t.randn(10, generator=g) * 0.5 + 1
    _check(
        compute_grpo_advantages(rewards, eps=0.5),
        solutions.compute_grpo_advantages(rewards, eps=0.5),
        [(solutions.compute_grpo_advantages(rewards), "Looks like you ignored the `eps` argument.")],
        "The advantages (with eps=0.5)",
    )

    # Large rewards, and rewards with a large offset compared to their spread (like the lengths of the samples), where
    # computing the variance as E[x^2] - E[x]^2 loses most of its precision (the rewards themselves are only accurate to
    # about 1e-4 here, hence the larger tolerance)
    for rewards, atol in [(t.randn(32, generator=g) * 1e4, 1e-5), (1000 + t.randn(32, generator=g), 1e-3)]:
        expected = solutions.compute_grpo_advantages(rewards)
        _check(compute_grpo_advantages(rewards), expected, [], "The advantages", atol=atol)

    # If every sample got the same reward, the advantages should be zero, not NaN
    advantages = compute_grpo_advantages(t.full((8,), 3.0))
    assert not advantages.isnan().any(), "Got NaN advantages when all rewards are equal. Did you add eps to the std?"
    t.testing.assert_close(advantages, t.zeros(8), check_dtype=False)

    print("All tests in `test_compute_grpo_advantages` passed!")


def test_get_logprobs(get_logprobs):
    solutions = _solutions()
    g = t.Generator().manual_seed(0)

    # Distinct, non-square dims (so mixing up dimensions shows up), and logits which cover all the positions, only the
    # last gen_len + 1 (as `get_logits` returns), or somewhere in between
    batch, seq_len, d_vocab, gen_len = 3, 11, 17, 5
    cases = [(gen_len, keep, 2.0) for keep in [seq_len, gen_len + 1, gen_len + 3]]
    # Edge cases: every token after the first is generated, a single generated token, and a batch of one
    cases += [(seq_len - 1, seq_len, 2.0), (1, seq_len, 2.0), (1, 2, 2.0)]
    # Numerical stability: very large logits (a naive log(softmax) gives log(0) = -inf for the unlikely tokens), large
    # offsets (a naive exp overflows), and ties
    cases += [(gen_len, seq_len, 1e4), (gen_len, gen_len + 1, "offset"), (gen_len, seq_len, "ties")]

    for i, (n_gen, keep, scale) in enumerate(cases):
        b = 1 if i == 5 else batch
        tokens = t.randint(0, d_vocab, (b, seq_len), generator=g)
        if scale == "offset":
            logits = t.randn(b, keep, d_vocab, generator=g) * 3 + 1e4
        elif scale == "ties":
            logits = t.zeros(b, keep, d_vocab)
        else:
            logits = t.randn(b, keep, d_vocab, generator=g) * scale
        logits_copy, tokens_copy = logits.clone(), tokens.clone()
        logits.requires_grad_(True)

        logprobs = get_logprobs(logits, tokens, n_gen)
        assert t.equal(logits.detach(), logits_copy) and t.equal(tokens, tokens_copy), (
            "get_logprobs shouldn't modify its inputs in place"
        )
        assert isinstance(logprobs, t.Tensor) and logprobs.shape == (b, n_gen), (
            f"Expected shape {(b, n_gen)} (with logits of shape {tuple(logits.shape)} and gen_len = {n_gen}), got "
            f"{tuple(getattr(logprobs, 'shape', ()))}"
        )
        assert logprobs.is_floating_point(), f"Expected float logprobs, got {logprobs.dtype}"
        assert not logprobs.isnan().any() and not logprobs.isinf().any(), (
            "Got NaN or inf logprobs: use `log_softmax`, which is numerically stable, rather than `softmax` then `log`."
        )

        ref_logits = logits_copy.clone().requires_grad_(True)
        expected = solutions.get_logprobs(ref_logits, tokens, n_gen)
        lp_all = logits_copy.log_softmax(-1)
        mistakes = []
        if scale != "ties":  # with all logits equal, all the mistakes below give the same values
            mistakes = [
                (
                    lp_all[:, -n_gen:].gather(-1, tokens[:, -n_gen:, None]).squeeze(-1),
                    "Off by one: the logits at position s predict the token at position s+1, so the generated tokens' "
                    "logprobs come from logits[:, -gen_len-1:-1].",
                ),
                (
                    lp_all[:, -n_gen - 1 : -1].gather(-1, tokens[:, -n_gen - 1 : -1, None]).squeeze(-1),
                    "Off by one: the generated tokens are tokens[:, -gen_len:] (the logits at position s predict the "
                    "token at position s+1).",
                ),
            ]
            if isinstance(scale, float) and scale < 100:
                mistakes.append(
                    (
                        logits_copy[:, -n_gen - 1 : -1].gather(-1, tokens[:, -n_gen:, None]).squeeze(-1),
                        "Looks like you returned raw logits. Take the log_softmax over the vocab first.",
                    )
                )
        # (With a large offset, the logits themselves are only accurate to about 1e-3)
        # Some of the mistakes can coincide with the correct values in the edge cases (e.g. a single generated token which
        # happens to equal the one before it), so we drop those
        atol = 2e-3 if scale == "offset" else 1e-5
        _check(logprobs, expected, mistakes, f"The logprobs (case {i})", atol=atol, drop_indistinct=True)

        # Gradients flow back to the logits (with random weights on each output, so a wrong gradient can't hide)
        assert logprobs.requires_grad, "The logprobs don't depend on the logits: did you detach something?"
        weights = t.randn(b, n_gen, generator=g)
        (logprobs * weights).sum().backward()
        (expected * weights).sum().backward()
        t.testing.assert_close(
            logits.grad,
            ref_logits.grad,
            atol=atol,
            rtol=1e-4,
            msg=lambda m: f"Wrong gradient wrt the logits (case {i}).\n{m}",
        )

    print("All tests in `test_get_logprobs` passed!")


def test_calc_kl_penalty_naive(calc_kl_penalty_naive):
    solutions = _solutions()
    g = t.Generator().manual_seed(0)
    batch_size, gen_len, d_vocab = 10, 5, 100
    kl_coef = 0.3  # not 0.5, so that forgetting kl_coef looks different from forgetting the 1/2 in k2

    # calc_kl_penalty_naive takes logprobs (not logits), so log-softmax first
    logprobs = t.randn(batch_size, gen_len, d_vocab, generator=g).log_softmax(-1)
    ref_logprobs = t.randn(batch_size, gen_len, d_vocab, generator=g).log_softmax(-1)

    kl_div = calc_kl_penalty_naive(logprobs, ref_logprobs, kl_coef)
    assert isinstance(kl_div, t.Tensor) and kl_div.numel() == 1
    kl_div_correct = solutions.calc_kl_penalty_naive(logprobs, ref_logprobs, kl_coef)
    if t.allclose(kl_div, kl_div_correct / kl_coef):
        raise Exception("Error: did you remember to multiply by the KL coefficient kl_coef?")
    t.testing.assert_close(kl_div, kl_div_correct)

    print("All tests in `test_calc_kl_penalty_naive` passed!")


def test_calc_kl_penalty_naive_stability(calc_kl_penalty_naive):
    solutions = _solutions()
    g = t.Generator().manual_seed(0)
    batch_size, gen_len, d_vocab = 10, 5, 1000
    kl_coef = 0.3

    # calc_kl_penalty_naive takes logprobs (not logits); log-softmax is the numerically-stable step
    logprobs = (t.randn(batch_size, gen_len, d_vocab, generator=g) * 100).log_softmax(-1)
    ref_logprobs = (t.randn(batch_size, gen_len, d_vocab, generator=g) * 100).log_softmax(-1)

    kl_div = calc_kl_penalty_naive(logprobs, ref_logprobs, kl_coef)
    assert not t.isnan(kl_div).any(), "Your KL divergence is NaN! Read the section on numerical stability for help."
    t.testing.assert_close(kl_div, solutions.calc_kl_penalty_naive(logprobs, ref_logprobs, kl_coef))

    print("All tests in `test_calc_kl_penalty_naive_stability` passed!")


def _check_kl_penalty(calc_kl_penalty, solutions, logprobs, ref_logprobs, kl_coef: float, name: str, edge_case=False):
    """Checks calc_kl_penalty's value and gradient (wrt logprobs) against the solution's, with hints for common mistakes."""
    batch_size, gen_len = logprobs.shape
    lp, ref = logprobs.clone().requires_grad_(True), ref_logprobs.clone()
    kl_div = calc_kl_penalty(lp, ref, kl_coef)
    assert t.equal(lp.detach(), logprobs) and t.equal(ref, ref_logprobs), "calc_kl_penalty shouldn't modify its inputs"
    assert isinstance(kl_div, t.Tensor) and kl_div.numel() == 1, "calc_kl_penalty should return a scalar tensor"
    kl_div = kl_div.reshape(())

    lp_ref = logprobs.clone().requires_grad_(True)
    expected = solutions.calc_kl_penalty(lp_ref, ref_logprobs, kl_coef)

    # Common mistakes: using the k1 or k3 estimator, the wrong sign of log_ratio in k3, dropping the 1/2, forgetting the
    # coefficient, summing rather than averaging, or the wrong sign
    log_ratio = logprobs - ref_logprobs
    mistakes = [
        (kl_coef * log_ratio.mean(), "Error: looks like you used k1 (log_ratio). Use k2 = 0.5 * log_ratio**2."),
        (
            kl_coef * ((-log_ratio).exp() - 1 + log_ratio).mean(),
            "Error: looks like you used k3. It's the best estimate of the KL's value, but its gradient is the gradient "
            "of the forward KL D_KL(pi_ref || pi_theta). Use k2 = 0.5 * log_ratio**2.",
        ),
        (
            kl_coef * (log_ratio.exp() - 1 - log_ratio).mean(),
            "Error: looks like you used k3 (with the log-ratio flipped). Use k2 = 0.5 * log_ratio**2.",
        ),
        (2 * expected, "Error: did you forget the factor of 1/2 in k2 = 0.5 * log_ratio**2?"),
        (expected / kl_coef, "Error: did you remember to multiply by the KL coefficient kl_coef?"),
        (expected * gen_len, "Error: looks like you summed over the sequence. Take the mean over batch and sequence."),
        (expected * batch_size, "Error: looks like you summed over the batch. Take the mean over batch and sequence."),
        (expected * batch_size * gen_len, "Error: looks like you summed. Take the mean over both batch and sequence."),
        (-expected, "Error: wrong sign. The KL penalty is nonnegative (we subtract it from the objective later)."),
    ]
    # (In the edge cases, some of these coincide with the correct value, e.g. summing over a single token, so we drop them)
    _check(kl_div, expected, mistakes, name, atol=1e-6, rtol=1e-5, drop_indistinct=edge_case)

    assert kl_div.requires_grad, "The KL penalty has no gradient with respect to logprobs: did you detach something?"
    kl_div.backward()
    expected.backward()
    assert lp.grad is not None, "No gradient reached logprobs"
    t.testing.assert_close(lp.grad, lp_ref.grad, msg=lambda m: f"{name}: wrong gradient wrt logprobs.\n{m}")


def test_calc_kl_penalty(calc_kl_penalty):
    solutions = _solutions()
    g = t.Generator().manual_seed(0)
    batch_size, gen_len, d_vocab = 7, 5, 50

    # Per-token logprobs of the sampled tokens, under the new and reference policies (shape (batch, gen_len)), from random
    # distributions over the vocab: both moderate ones, and very peaked ones (large logits) whose log-ratios are large.
    # The kl_coefs are not 0.5, so that forgetting kl_coef looks different from forgetting the 1/2 in k2.
    for kl_coef, scale in [(0.3, 1.0), (1.7, 1.0), (0.3, 30.0)]:
        tokens = t.randint(0, d_vocab, (batch_size, gen_len), generator=g)
        logprobs = _sampled_logprobs(t.randn(batch_size, gen_len, d_vocab, generator=g) * scale, tokens)
        ref_logprobs = _sampled_logprobs(t.randn(batch_size, gen_len, d_vocab, generator=g) * scale, tokens)
        _check_kl_penalty(calc_kl_penalty, solutions, logprobs, ref_logprobs, kl_coef, f"The KL penalty ({kl_coef=})")

    print("All tests in `test_calc_kl_penalty` passed!")


def test_calc_kl_penalty_stability(calc_kl_penalty):
    solutions = _solutions()
    g = t.Generator().manual_seed(0)
    batch_size, gen_len = 10, 5
    kl_coef = 0.3

    # Individually tiny probabilities (very negative logprobs), but moderate per-token log-ratios.
    # A numerically stable implementation works with the log-ratio `logprobs - ref_logprobs`
    # directly, so these are fine. An implementation that exponentiates the two logprobs separately
    # (e.g. forming the ratio as `logprobs.exp() / ref_logprobs.exp()`) underflows to 0 / 0 -> NaN.
    ref_logprobs = t.full((batch_size, gen_len), -1000.0)
    logprobs = ref_logprobs + t.randn(batch_size, gen_len, generator=g)
    # Sanity-check the premise: exponentiating the logprobs individually really does underflow
    assert (logprobs.exp() == 0).all() and (ref_logprobs.exp() == 0).all()

    # Then logprobs near -1e4 under one policy but not the other (a huge log-ratio, so exp(log-ratio) underflows or
    # overflows, although k2 is just a large finite number), and a batch of a single token
    huge = t.full((batch_size, gen_len), -1e4) + t.randn(batch_size, gen_len, generator=g)
    moderate = -t.rand(batch_size, gen_len, generator=g) * 3
    cases = [(logprobs, ref_logprobs), (huge, moderate), (moderate, huge), (huge[:1, :1], moderate[:1, :1])]

    for i, (lp, ref) in enumerate(cases):
        kl_div = calc_kl_penalty(lp.clone(), ref.clone(), kl_coef)
        assert t.isfinite(kl_div).all(), (
            f"Your KL divergence is {kl_div.item()} (case {i})! Work with the log-ratio (logprobs - ref_logprobs) "
            "directly, rather than exponentiating logprobs and ref_logprobs separately (which underflows to 0/0). Read "
            "the section on numerical stability for help."
        )
        _check_kl_penalty(calc_kl_penalty, solutions, lp, ref, kl_coef, f"The KL penalty (stability case {i})", True)

    print("All tests in `test_calc_kl_penalty_stability` passed!")


def _clipped_surrogate_mistakes(logprobs, old_logprobs, advantages, clip_coef, expected) -> list:
    """Values of calc_clipped_surrogate_objective under common mistakes, with a hint for each."""
    batch, gen_len = logprobs.shape
    eps = clip_coef
    ratio = (logprobs - old_logprobs).exp()
    adv = advantages[:, None]
    clipped = ratio.clip(1 - eps, 1 + eps)
    normalized = ((advantages - advantages.mean()) / (advantages.std() + 1e-8))[:, None]
    return [
        (
            t.minimum(ratio * normalized, clipped * normalized).mean(),
            "Looks like you normalized the advantages. They're already normalized within each group, so don't "
            "normalize them again here.",
        ),
        (
            t.maximum(ratio * adv, clipped * adv).mean(),
            "Looks like you took the maximum of the clipped and unclipped terms, rather than the minimum.",
        ),
        ((ratio * adv).mean(), "Looks like you didn't clip the ratio."),
        (
            (clipped * adv).mean(),
            "Looks like you only used the clipped term. Take the minimum of the clipped and unclipped terms.",
        ),
        (
            t.minimum(ratio * adv, ratio.clip(max=1 + eps) * adv).mean(),
            "Looks like you only clipped the ratio from above. Clip it to [1 - clip_coef, 1 + clip_coef].",
        ),
        (
            t.minimum(ratio * adv, ratio.clip(min=1 - eps) * adv).mean(),
            "Looks like you only clipped the ratio from below. Clip it to [1 - clip_coef, 1 + clip_coef].",
        ),
        (
            t.minimum(ratio * adv, (ratio * adv).clip(1 - eps, 1 + eps)).mean(),
            "Looks like you clipped ratio * advantage. Clip the ratio itself, then multiply by the advantage.",
        ),
        (
            t.minimum(ratio * adv, (logprobs - old_logprobs).clip(-eps, eps).exp() * adv).mean(),
            "Looks like you clipped the log-ratio to [-clip_coef, clip_coef]. Clip the ratio to "
            "[1 - clip_coef, 1 + clip_coef].",
        ),
        (
            t.minimum(1 / ratio * adv, (1 / ratio).clip(1 - eps, 1 + eps) * adv).mean(),
            "Your ratio is upside down: it should be pi_theta / pi_theta_old = exp(logprobs - old_logprobs).",
        ),
        (expected * batch * gen_len, "Looks like you summed over all tokens. Take the mean over batch and sequence."),
        (expected * batch, "Looks like you summed over the batch. Take the mean over batch and sequence."),
        (expected * gen_len, "Looks like you summed over the sequence. Take the mean over batch and sequence."),
        (-expected, "Wrong sign: return the objective to maximize, not a loss."),
    ]


def _check_clipped_surrogate(
    fn, solutions, logprobs, old_logprobs, advantages, clip_coef, name, hints=True, edge_case=False
):
    """Checks calc_clipped_surrogate_objective's value and gradient (wrt logprobs) against the solution's."""
    lp, old, adv = logprobs.clone().requires_grad_(True), old_logprobs.clone(), advantages.clone()
    objective = fn(lp, old, adv, clip_coef)
    assert t.equal(lp.detach(), logprobs) and t.equal(old, old_logprobs) and t.equal(adv, advantages), (
        "calc_clipped_surrogate_objective shouldn't modify its inputs in place"
    )
    assert isinstance(objective, t.Tensor) and objective.numel() == 1, "Should return a scalar tensor"
    assert objective.is_floating_point(), f"Expected a float objective, got {objective.dtype}"
    objective = objective.reshape(())

    lp_ref = logprobs.clone().requires_grad_(True)
    expected = solutions.calc_clipped_surrogate_objective(lp_ref, old_logprobs, advantages, clip_coef)
    mistakes = _clipped_surrogate_mistakes(logprobs, old_logprobs, advantages, clip_coef, expected) if hints else []
    # (In the extreme cases, a few huge terms dominate the mean, so some mistakes only change the value by a tiny relative
    # amount: we drop the hints for those, and rely on the gradient check below to catch them)
    _check(objective, expected, mistakes, name, atol=1e-6, rtol=1e-5, drop_indistinct=edge_case)

    # The gradient is where clipping really matters: it's zero for the tokens whose ratio has been clipped
    assert objective.requires_grad, "The objective has no gradient with respect to logprobs: did you detach something?"
    objective.backward()
    expected.backward()
    assert lp.grad is not None, "No gradient reached logprobs"
    t.testing.assert_close(lp.grad, lp_ref.grad, msg=lambda m: f"{name}: wrong gradient wrt logprobs.\n{m}")


def test_calc_clipped_surrogate_objective(calc_clipped_surrogate_objective):
    solutions = _solutions()
    g = t.Generator().manual_seed(0)
    batch, gen_len, d_vocab = 13, 7, 50

    # Logprobs of sampled tokens under random distributions over the vocab, from moderate to very peaked (large logits).
    # The old policy is a perturbed copy of the new one, so the ratios spread well beyond [1 - eps, 1 + eps] in both
    # directions. The advantages have both signs, a nonzero mean and non-unit std (so re-normalizing them would change
    # the answer).
    for clip_coef, scale, noise in [(0.2, 1.0, 0.4), (0.1, 1.0, 0.4), (0.3, 20.0, 0.1), (0.2, 100.0, 0.05)]:
        tokens = t.randint(0, d_vocab, (batch, gen_len), generator=g)
        logits = t.randn(batch, gen_len, d_vocab, generator=g) * scale
        old_logits = logits * (1 + noise * t.randn(batch, gen_len, 1, generator=g))
        old_logits = old_logits + t.randn(batch, gen_len, d_vocab, generator=g) * noise * scale
        logprobs, old_logprobs = _sampled_logprobs(logits, tokens), _sampled_logprobs(old_logits, tokens)
        advantages = t.randn(batch, generator=g) * 1.5 + 0.3

        # Make sure every case of the clipping is covered: ratio above and below the clip range, for each sign of the
        # advantage (plus some tokens inside the clip range, except for the most peaked distributions)
        ratio, adv = (logprobs - old_logprobs).exp(), advantages[:, None].expand(batch, gen_len)
        cases = [ratio > 1 + clip_coef, ratio < 1 - clip_coef] + ([(ratio - 1).abs() < clip_coef] if scale < 100 else [])
        for case in cases:
            for sign in [adv > 0, adv < 0]:
                assert (case & sign).any(), "Test setup error: not every case of the clipping is covered"

        _check_clipped_surrogate(
            calc_clipped_surrogate_objective,
            solutions,
            logprobs,
            old_logprobs,
            advantages,
            clip_coef,
            f"The clipped surrogate objective ({clip_coef=}, logit scale {scale})",
            edge_case=scale > 1,
        )

    # At the start of a learning phase (ratio = 1), the gradient should be the policy gradient: A_t * grad log pi
    old_logprobs = -t.rand(batch, gen_len, generator=g) * 5
    advantages = t.randn(batch, generator=g)
    logprobs = old_logprobs.clone().requires_grad_(True)
    objective = calc_clipped_surrogate_objective(logprobs, old_logprobs, advantages, 0.2)
    assert objective.requires_grad, "The objective has no gradient with respect to logprobs: did you detach something?"
    objective.backward()
    t.testing.assert_close(logprobs.grad, advantages[:, None].expand(batch, gen_len) / (batch * gen_len))

    # A single sequence with a single token, on each side of the clip range
    for log_ratio, a in [(0.5, 1.0), (0.5, -1.0), (-0.5, 1.0), (-0.5, -1.0)]:
        old_logprobs = t.tensor([[-2.0]])
        _check_clipped_surrogate(
            calc_clipped_surrogate_objective,
            solutions,
            old_logprobs + log_ratio,
            old_logprobs,
            t.tensor([a]),
            0.2,
            f"The clipped surrogate objective (one token, log-ratio {log_ratio}, advantage {a})",
            hints=False,
        )

    print("All tests in `test_calc_clipped_surrogate_objective` passed!")


def test_calc_clipped_surrogate_objective_stability(calc_clipped_surrogate_objective):
    """
    Not called from the notebook. Very negative logprobs (so exponentiating them separately underflows to 0 / 0) with
    moderate log-ratios: computing the ratio as exp(logprobs - old_logprobs) is stable.
    """
    solutions = _solutions()
    g = t.Generator().manual_seed(0)
    batch, gen_len = 6, 4
    old_logprobs = -1e4 + t.randn(batch, gen_len, generator=g)
    logprobs = old_logprobs + t.randn(batch, gen_len, generator=g) * 0.5
    advantages = t.randn(batch, generator=g)
    objective = calc_clipped_surrogate_objective(logprobs, old_logprobs, advantages, 0.2)
    assert t.isfinite(objective).all(), (
        f"Got {objective.item()} for very negative logprobs: compute the ratio as exp(logprobs - old_logprobs), rather "
        "than exponentiating each logprob separately (which underflows to 0 / 0)."
    )
    _check_clipped_surrogate(
        calc_clipped_surrogate_objective, solutions, logprobs, old_logprobs, advantages, 0.2, "The objective", hints=False
    )
    print("All tests in `test_calc_clipped_surrogate_objective_stability` passed!")


def test_reward_fn_sentiment_imdb(reward_fn_sentiment_imdb):
    samples = ["This movie was wonderful, I loved every minute of it.", "This movie was awful, a total waste of time."]
    pos = reward_fn_sentiment_imdb(samples, direction="pos")
    neg = reward_fn_sentiment_imdb(samples, direction="neg")
    assert pos.shape == neg.shape == (2,), f"Expected a reward for each sample, got shape {tuple(pos.shape)}"
    assert pos.dtype in (t.float32, t.float64), (
        f"Expected float32 rewards (cast the logits to float for the softmax), got {pos.dtype}"
    )
    # On the same device as the classifier (the trainer computes the advantages on the policy's device)
    device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
    assert pos.device.type == device.type, f"The rewards should be on {device}, got {pos.device}"
    assert ((pos >= 0) & (pos <= 1)).all(), "Rewards should be probabilities (did you forget the softmax?)"
    t.testing.assert_close(pos + neg, t.ones_like(pos), atol=1e-3, rtol=0)
    assert pos[0] > 0.9 and pos[1] < 0.1, f"Unexpected positive-sentiment rewards {pos.tolist()}: check the class index"

    # A sample's reward shouldn't depend on the rest of the batch: batched with a longer sample, a short one gets padded,
    # and the classifier should ignore the padding (pass it the tokenizer's attention mask, not just the input ids). This
    # sample's reward isn't saturated near 0 or 1, so it's sensitive to the padding.
    short, long = "It was fine I guess.", samples[0] + " The acting, the music and the story were all brilliant."
    alone, batched = reward_fn_sentiment_imdb([short]), reward_fn_sentiment_imdb([long, short])
    t.testing.assert_close(
        batched[1:],
        alone,
        atol=2e-2,
        rtol=0,
        msg="A short sample's reward changes when it's batched with a longer one: did you pass the attention mask?",
    )

    # Samples longer than the classifier's 512-token context should be truncated, not crash
    try:
        long = reward_fn_sentiment_imdb(["This movie was wonderful. " * 200, "Short and awful."], direction="pos")
    except Exception as e:
        raise Exception(f"Failed on a sample longer than 512 tokens: did you pass `truncation=True`? ({e!r})") from e
    assert long.shape == (2,), f"Expected a reward for each sample, got shape {tuple(long.shape)}"
    print("All tests in `test_reward_fn_sentiment_imdb` passed!")


@contextlib.contextmanager
def _tiny_model_dir(arch: str):
    """A temporary directory holding a tiny model (see `_save_tiny_model`), without the loading progress bars."""
    from transformers.utils import logging

    progress_bar = logging.is_progress_bar_enabled()
    logging.disable_progress_bar()
    try:
        with tempfile.TemporaryDirectory() as model_dir:
            _save_tiny_model(model_dir, arch)
            yield model_dir
    finally:
        if progress_bar:
            logging.enable_progress_bar()


def _save_tiny_model(path: str, arch: str = "gpt2") -> None:
    """
    Saves a tiny randomly-initialized language model (with GPT-2's tokenizer) to `path`, which `from_pretrained` can then
    load. It's much faster than a real model, and it barely uses any GPU memory. Its weights are initialized with a large
    std, so that its next-token distributions are fairly peaked (rather than almost uniform).
    """
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    special = dict(bos_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
    with _fork_rng():
        t.manual_seed(0)
        if arch == "gpt2":
            config = GPT2Config(
                vocab_size=len(tokenizer), n_positions=64, n_embd=32, n_layer=2, n_head=2, initializer_range=0.5, **special
            )
            model = GPT2LMHeadModel(config)
        else:
            config = LlamaConfig(
                vocab_size=len(tokenizer),
                hidden_size=32,
                intermediate_size=48,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=64,
                initializer_range=0.5,
                **special,
            )
            model = LlamaForCausalLM(config)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def test_grpo_trainer(GRPO_Trainer, GRPO_Args):
    """
    Tests the three methods of GRPO_Trainer on a tiny randomly-initialized GPT-2, with a policy that's been changed so
    it's different from the reference model (so we can tell which model each set of logprobs came from).
    """
    # Full-precision float32 matmuls for this test (the exercises use TF32, whose results depend slightly on the batch
    # size, enough to blur the comparisons below)
    precision = t.get_float32_matmul_precision()
    t.set_float32_matmul_precision("highest")
    try:
        with _tiny_model_dir("gpt2") as model_dir, _fork_rng():
            _test_grpo_trainer(GRPO_Trainer, GRPO_Args, model_dir)
    finally:
        t.set_float32_matmul_precision(precision)
    gc.collect()
    if t.cuda.is_available():
        t.cuda.empty_cache()
    print("All tests in `test_grpo_trainer` passed!")


def _make_reward_fn(get_device):
    """
    A reward function giving each sample a pseudo-random reward, determined by its text (so calling it twice on the same
    samples gives the same rewards). The rewards have a nonzero mean and non-unit std, and they're very unlikely to all be
    equal (which would make the advantages degenerate).
    """

    def reward_fn(samples):
        return t.tensor([zlib.crc32(s.encode()) % 1000 / 100 for s in samples], device=get_device())

    return reward_fn


def _check_rollout(trainer, memory, policy_lp, ref_lp, grad_modes, atol=1e-4):
    """Checks the output of `trainer.rollout_phase`, given the expected logprobs under the policy and reference model."""
    solutions = _solutions()
    args = trainer.args
    gen_len = args.gen_len
    assert not any(grad_modes), (
        "A forward pass during rollout_phase ran with gradients enabled: do the whole rollout under t.no_grad()"
    )
    for name in ["sample_ids", "logprobs", "ref_logprobs", "advantages"]:
        assert hasattr(memory, name), f"rollout_phase should return a ReplayMemory (missing `{name}`)"
    for name in ["logprobs", "ref_logprobs", "advantages"]:
        assert not getattr(memory, name).requires_grad, f"memory.{name} has gradients: do the rollout under t.no_grad()"
    assert memory.sample_ids.shape[0] == args.batch_size
    for name in ["logprobs", "ref_logprobs"]:
        value = getattr(memory, name)
        assert value.shape == (args.batch_size, gen_len), (
            f"memory.{name} should have shape {(args.batch_size, gen_len)}, got {tuple(value.shape)}"
        )
        assert value.dtype in (t.float32, t.float64), f"memory.{name} should be float32, got {value.dtype}"
    device = memory.sample_ids.device
    for name in ["logprobs", "ref_logprobs", "advantages"]:
        value = getattr(memory, name)
        assert value.device == device, f"memory.{name} is on {value.device}, but the samples are on {device}"

    # Logprobs under the current policy and the reference model
    assert (policy_lp - ref_lp).abs().max() > 20 * atol, "Test setup error: the policy and reference model agree"
    if t.allclose(memory.logprobs, ref_lp, atol=atol):
        raise Exception("memory.logprobs look like the reference model's: they should be under the current policy.")
    t.testing.assert_close(
        memory.logprobs.to(policy_lp.dtype), policy_lp, atol=atol, rtol=1e-4, msg="memory.logprobs are wrong"
    )
    if t.allclose(memory.ref_logprobs, policy_lp, atol=atol):
        raise Exception("memory.ref_logprobs look like the policy's: use `self.get_ref_logprobs`.")
    t.testing.assert_close(
        memory.ref_logprobs.to(ref_lp.dtype), ref_lp, atol=atol, rtol=1e-4, msg="memory.ref_logprobs are wrong"
    )

    # The advantages come from the rewards of the full samples (prompt + completion)
    completions = trainer.tokenizer.batch_decode(memory.sample_ids[:, -gen_len:], skip_special_tokens=True)
    rewards = args.reward_fn([args.prompt + c for c in completions]).to(memory.advantages.device)
    assert rewards.std() > 0.1, "Test setup error: the rewards are all equal"
    expected = solutions.compute_grpo_advantages(rewards)
    if memory.advantages.shape == rewards.shape and not t.allclose(memory.advantages, expected, atol=1e-5, rtol=1e-5):
        if t.allclose(memory.advantages, rewards):
            raise Exception("memory.advantages are the raw rewards: compute the GRPO advantages from them.")
        completion_rewards = args.reward_fn(completions).to(memory.advantages.device)
        if t.allclose(memory.advantages, solutions.compute_grpo_advantages(completion_rewards), atol=1e-5, rtol=1e-5):
            raise Exception("Call the reward function on the samples (prompt + completion), not just the completions.")
    t.testing.assert_close(
        memory.advantages.to(expected.dtype), expected, atol=1e-5, rtol=1e-5, msg="memory.advantages are wrong"
    )


def _record_grad_modes(*models):
    """Adds forward hooks to `models` recording whether each forward pass runs with gradients enabled."""
    grad_modes = []
    hook = lambda *_: grad_modes.append(t.is_grad_enabled())  # noqa: E731
    handles = [m.register_forward_pre_hook(hook) for m in models if m is not None]
    return grad_modes, handles


def _test_grpo_trainer(GRPO_Trainer, GRPO_Args, model_dir):
    solutions = _solutions()
    g = t.Generator().manual_seed(0)
    trainer = None
    reward_fn = _make_reward_fn(lambda: next(trainer.model.parameters()).device)

    # Non-default clip_coef and kl_coef (so hardcoding either shows up), and distinct batch_size, minibatch_size and gen_len
    args = GRPO_Args(
        base_model=model_dir,
        prompt="One two three",
        batch_size=12,
        num_minibatches=3,
        batches_per_learning_phase=2,
        gen_len=5,
        total_phases=10,  # more phases than minibatches, so stepping the scheduler too often shows up below
        warmup_steps=0,
        lr=1e-4,
        clip_coef=0.15,
        kl_coef=0.4,
        max_grad_norm=0.2,  # small enough that the gradients are always clipped (we check this below)
        reward_fn=reward_fn,
    )
    trainer = GRPO_Trainer(args)
    model, ref_model = trainer.model, trainer.ref_model
    device = next(model.parameters()).device
    gen_len = args.gen_len
    params = [p for p in model.parameters() if p.requires_grad]

    # Make the policy different from the reference model (like a temperature change)
    with t.no_grad():
        model.transformer.ln_f.weight.mul_(1.5)

    @t.no_grad()
    def logprobs_under(m, sample_ids):
        return solutions.get_logprobs(solutions.get_logits(m, sample_ids, gen_len), sample_ids, gen_len)

    def solution_objective(minibatch):
        sample_ids = minibatch.sample_ids
        logprobs = solutions.get_logprobs(solutions.get_logits(model, sample_ids, gen_len), sample_ids, gen_len)
        surrogate = solutions.calc_clipped_surrogate_objective(
            logprobs, minibatch.logprobs, minibatch.advantages, args.clip_coef
        )
        return surrogate - solutions.calc_kl_penalty(logprobs, minibatch.ref_logprobs, args.kl_coef)

    # ==================== rollout_phase ====================
    trainer.step, trainer.phase = 0, 0
    weights_before = [p.detach().clone() for p in model.parameters()]
    grad_modes, handles = _record_grad_modes(model, ref_model)
    try:
        memory = trainer.rollout_phase()
    finally:
        for h in handles:
            h.remove()
    assert all(t.equal(p, w) for p, w in zip(model.parameters(), weights_before)), "rollout_phase changed the model"
    _check_rollout(
        trainer,
        memory,
        logprobs_under(model, memory.sample_ids),
        logprobs_under(ref_model, memory.sample_ids),
        grad_modes,
    )
    # A second rollout should give new samples (so rollout_phase shouldn't reseed the RNG, for example)
    with contextlib.redirect_stdout(io.StringIO()):
        memory_2 = trainer.rollout_phase()
    assert not t.equal(memory_2.sample_ids, memory.sample_ids), (
        "Two rollouts gave exactly the same samples: is rollout_phase reseeding the random number generator?"
    )
    print("`rollout_phase` passed!")

    # ==================== compute_grpo_objective ====================
    # A synthetic minibatch, where the logprobs at rollout time (old) and under the reference model both differ from the
    # policy's current logprobs, so the ratios land well outside [1 - eps, 1 + eps] on both sides (for both signs of the
    # advantage), and the KL penalty is nonzero
    mb_size, seq_len = args.minibatch_size, 9
    sample_ids = t.randint(0, model.config.vocab_size, (mb_size, seq_len), generator=g).to(device)
    logprobs = logprobs_under(model, sample_ids)
    old_logprobs = logprobs + (t.randn(mb_size, gen_len, generator=g) * 0.5).to(device)
    ref_logprobs = logprobs + (t.randn(mb_size, gen_len, generator=g) * 0.5).to(device)
    advantages = ((t.rand(mb_size, generator=g) + 0.5) * t.tensor([1.0, -1.0, 1.0, -1.0])).to(device)
    ratio, adv = (logprobs - old_logprobs).exp(), advantages[:, None]
    for clipped in [ratio > 1 + args.clip_coef, ratio < 1 - args.clip_coef]:
        assert (clipped & (adv > 0)).any() and (clipped & (adv < 0)).any(), "Test setup error: clipping not covered"
    minibatch = solutions.ReplayMinibatch(sample_ids, old_logprobs, ref_logprobs, advantages)
    inputs = [x.clone() for x in (sample_ids, old_logprobs, ref_logprobs, advantages)]

    objective = trainer.compute_grpo_objective(minibatch)
    assert all(t.equal(x, y) for x, y in zip(inputs, (sample_ids, old_logprobs, ref_logprobs, advantages))), (
        "compute_grpo_objective shouldn't modify the minibatch"
    )
    assert isinstance(objective, t.Tensor) and objective.numel() == 1, "compute_grpo_objective should return a scalar"
    assert objective.is_floating_point(), f"The objective should be a float tensor, got {objective.dtype}"
    assert objective.requires_grad, "The objective has no gradient: don't compute the policy's logprobs under no_grad"
    expected = solution_objective(minibatch)

    surrogate = solutions.calc_clipped_surrogate_objective(logprobs, old_logprobs, advantages, args.clip_coef)
    kl = solutions.calc_kl_penalty(logprobs, ref_logprobs, args.kl_coef)
    mistakes = [
        (surrogate + kl, "Wrong sign on the KL penalty: we maximize the objective, so it's surrogate - kl_penalty."),
        (surrogate, "Looks like you left out the KL penalty."),
        (
            solutions.calc_clipped_surrogate_objective(logprobs, ref_logprobs, advantages, args.clip_coef) - kl,
            "Looks like you used ref_logprobs in the importance ratio. The ratio should be against the policy at rollout "
            "time, i.e. minibatch.logprobs.",
        ),
        (
            surrogate - solutions.calc_kl_penalty(logprobs, old_logprobs, args.kl_coef),
            "Looks like you used minibatch.logprobs in the KL penalty. It should be against the reference policy, i.e. "
            "minibatch.ref_logprobs.",
        ),
        (kl - surrogate, "Looks like you returned a loss (the negative objective). We maximize the objective."),
        (
            solutions.calc_clipped_surrogate_objective(logprobs, old_logprobs, advantages, 0.2) - kl,
            "Use self.args.clip_coef, rather than a hardcoded clip coefficient.",
        ),
        (
            surrogate - solutions.calc_kl_penalty(logprobs, ref_logprobs, 0.3),
            "Use self.args.kl_coef, rather than a hardcoded KL coefficient.",
        ),
    ]
    _check(objective, expected, mistakes, "The output of compute_grpo_objective", atol=1e-5, rtol=1e-4)

    # The gradient wrt the model's parameters should match too (this is what the optimizer uses)
    grads = t.autograd.grad(objective, params, allow_unused=True)
    expected_grads = t.autograd.grad(expected, params)
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    for name, grad, expected_grad in zip(names, grads, expected_grads):
        assert grad is not None, f"The objective doesn't depend on {name}: compute the logprobs with the model"
        # Relative to the size of the whole gradient (elementwise tolerances are too strict for float32 noise here)
        error = (grad.to(expected_grad.dtype) - expected_grad).norm()
        assert error <= 1e-3 * expected_grad.norm() + 1e-8, (
            f"Wrong gradient of the objective wrt {name} (relative error {error / expected_grad.norm():.2g})"
        )
    print("`compute_grpo_objective` passed!")

    # ==================== learning_phase ====================
    # A copy of the trainer, using the solution's learning_phase, to compare against at the end
    ref_trainer = solutions.GRPO_Trainer(args)
    ref_trainer.model.load_state_dict(model.state_dict())
    # A nonzero starting step (as in a later phase), so resetting self.step shows up
    ref_trainer.step = trainer.step = 5
    model.zero_grad(set_to_none=True)

    memory_lp_before, synthetic_lp_before = logprobs_under(model, memory.sample_ids), logprobs_under(model, sample_ids)

    # Use the same minibatches for both trainers
    minibatches = memory.get_minibatches()
    memory.get_minibatches = lambda: minibatches

    # Record what happens inside the learning phase: the objectives, and the gradients at each optimizer step (which
    # should be the gradient of that minibatch's objective, clipped to max_grad_norm)
    objectives, n_steps, prev_grad, current, steps = [], [0], [None], [None], []
    lr_start = trainer.optimizer.param_groups[0]["lr"]
    compute_objective, optimizer_step = trainer.compute_grpo_objective, trainer.optimizer.step

    def recording_objective(mb):
        out = compute_objective(mb)
        objectives.append(out.item())
        current[0] = mb
        return out

    def flat(tensors):
        return t.cat([x.flatten() for x in tensors])

    def matches(a, b):
        return (a - b).norm() <= 1e-4 * b.norm()

    # functools.wraps keeps the LR scheduler's marker on `step`, so it doesn't warn that `step` was overridden
    @functools.wraps(optimizer_step)
    def recording_step(*a, **kw):
        n_steps[0] += 1
        steps.append(trainer.step)
        assert any(p.grad is not None for p in params), (
            "No gradients when stepping the optimizer: did you call `objective.backward()`?"
        )
        assert current[0] is not None, "The optimizer was stepped before computing the objective"
        grad = flat([t.zeros_like(p) if p.grad is None else p.grad for p in params])
        raw = flat(t.autograd.grad(solution_objective(current[0]), params))
        assert raw.norm() > 2 * args.max_grad_norm, "Test setup error: the gradients aren't clipped"
        clipped = raw * min(1.0, args.max_grad_norm / (raw.norm().item() + 1e-6))
        if not matches(grad, clipped):
            if matches(grad, raw):
                raise Exception(
                    f"Gradient norm at the optimizer step was {grad.norm():.3g}, but max_grad_norm = "
                    f"{args.max_grad_norm}: clip the gradients (after backward, before the optimizer step)"
                )
            if matches(grad, -clipped):
                raise Exception(
                    "The gradients have the wrong sign: call backward on the objective (not on its negative). The "
                    "optimizer is set up with maximize=True, so it does gradient ascent."
                )
            if grad.norm() > args.max_grad_norm * 1.01:
                raise Exception(
                    f"Gradient norm at the optimizer step was {grad.norm():.3g}, but max_grad_norm = "
                    f"{args.max_grad_norm}: clip the total gradient norm to self.args.max_grad_norm (after backward, "
                    "before the optimizer step)"
                )
            accumulated = None if prev_grad[0] is None else prev_grad[0] + raw
            if accumulated is not None and matches(grad / grad.norm(), accumulated / accumulated.norm()):
                raise Exception(
                    "The gradients at the optimizer step aren't just this minibatch's: zero them before each backward "
                    "pass (otherwise they accumulate across minibatches)"
                )
            raise Exception(
                "The gradients at the optimizer step don't match the (clipped) gradient of this minibatch's objective"
            )
        lr = trainer.optimizer.param_groups[0]["lr"]
        assert lr == lr_start, (
            f"The learning rate changed during the learning phase (from {lr_start:.3g} to {lr:.3g}): step the scheduler "
            "once per learning phase, after all the optimizer steps"
        )
        prev_grad[0] = grad.clone()
        current[0] = None
        return optimizer_step(*a, **kw)

    trainer.compute_grpo_objective = recording_objective
    trainer.optimizer.step = recording_step
    try:
        result = trainer.learning_phase(memory)
    finally:
        trainer.compute_grpo_objective, trainer.optimizer.step = compute_objective, optimizer_step

    n_minibatches = args.num_minibatches * args.batches_per_learning_phase
    assert len(objectives) == n_minibatches, (
        f"Expected the objective to be computed once per minibatch ({n_minibatches} times), got {len(objectives)}"
    )
    assert n_steps[0] == n_minibatches, f"Expected one optimizer step per minibatch ({n_minibatches}), got {n_steps[0]}"
    assert trainer.step == 5 + n_minibatches and all(b - a == 1 for a, b in zip(steps, steps[1:])), (
        f"self.step should be incremented by 1 for each minibatch: starting from 5, we expected 5 + {n_minibatches}, and "
        f"got {trainer.step} (with self.step = {steps} at the optimizer steps)"
    )
    assert trainer.scheduler.last_epoch == 1, (
        f"The scheduler was stepped {trainer.scheduler.last_epoch} times: step it once per learning phase"
    )
    mean_objective = sum(objectives) / len(objectives)
    assert not (isinstance(result, t.Tensor) and result.requires_grad), (
        "learning_phase should return a float (e.g. with `.item()`), not a tensor that's still attached to the graph"
    )
    assert abs(float(result) - mean_objective) < 1e-5, (
        f"learning_phase should return the mean objective over the minibatches ({mean_objective:.5f}), got {result}"
    )

    # Finally, the solution's learning phase (from the same starting point, on the same minibatches) should end up with
    # the same model, and the reference model shouldn't have changed. We compare the models' logprobs rather than their
    # weights: some weights (like the attention's key biases) have a gradient of exactly zero mathematically, so their
    # gradient is just rounding noise, which Adam scales up to updates of size lr in an arbitrary direction.
    ref_result = ref_trainer.learning_phase(memory)
    del memory.get_minibatches
    assert abs(float(result) - ref_result) < 1e-3, f"learning_phase returned {result}, expected {ref_result}"
    assert any(not t.equal(p, w) for p, w in zip(model.parameters(), weights_before)), "The model's weights didn't change"
    for ids, lp_before in [(memory.sample_ids, memory_lp_before), (sample_ids, synthetic_lp_before)]:
        lp, lp_ref = logprobs_under(model, ids), logprobs_under(ref_trainer.model, ids)
        assert (lp_ref - lp_before).abs().max() > 0.05, "Test setup error: the learning phase barely changed the model"
        t.testing.assert_close(
            lp, lp_ref, atol=2e-3, rtol=0, msg=lambda m: f"After learning_phase, the model's logprobs are wrong.\n{m}"
        )
    for (name, p), p_ref in zip(ref_model.named_parameters(), ref_trainer.ref_model.parameters()):
        assert t.equal(p, p_ref), f"The reference model's weights ({name}) changed: it should be frozen"
    print("`learning_phase` passed!")
