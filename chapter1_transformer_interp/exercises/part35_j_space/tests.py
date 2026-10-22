import math
import sys
from pathlib import Path
from typing import Callable

import torch

if str(exercises_dir := Path(__file__).parent.parent) not in sys.path:
    sys.path.append(str(exercises_dir))


def test_lens_logits(lens_logits: Callable, unembed, nonsem: torch.Tensor):
    torch.manual_seed(0)
    d = unembed.W_U.shape[1]
    h = torch.randn(3, d, device=unembed.W_U.device) * 50
    J = {5: torch.randn(d, d, device=h.device) / d**0.5}

    out = lens_logits(h, 5, None, filter_nonsemantic=False)
    assert out.shape == (3, unembed.W_U.shape[0]), f"Expected shape (3, d_vocab), got {tuple(out.shape)}"
    assert out.dtype == torch.float32, "Readout should be in float32"
    torch.testing.assert_close(out, unembed(h), msg="Logit lens (J=None) should just apply the unembedding")

    out_J = lens_logits(h, 5, J, filter_nonsemantic=False)
    torch.testing.assert_close(out_J, unembed(h @ J[5].T), msg="J-lens readout should be unembed(h @ J[layer].T)")

    out_f = lens_logits(h, 5, J, filter_nonsemantic=True)
    assert torch.isinf(out_f[:, nonsem]).all() and (out_f[:, nonsem] < 0).all(), "Non-semantic tokens should be -inf"
    torch.testing.assert_close(out_f[:, ~nonsem], out_J[:, ~nonsem])
    print("All tests in `test_lens_logits` passed!")


def test_jacobian_for_prompt(jacobian_for_prompt: Callable, model, input_ids, unembed):
    """Checks a few rows of the estimator against direct VJPs (one backward pass per row)."""
    from part35_j_space import utils

    layers = utils.get_layers(model)
    src, tgt, skip = len(layers) - 4, len(layers) - 2, 4
    J = jacobian_for_prompt(model, input_ids, [src], tgt, dim_batch=8, skip_first=skip)
    d = unembed.W_U.shape[1]
    assert set(J) == {src}, f"Expected keys {{{src}}}, got {set(J)}"
    assert J[src].shape == (d, d), f"Expected shape ({d}, {d}), got {tuple(J[src].shape)}"

    seq = input_ids.shape[1]
    valid = torch.arange(skip, seq - 1, device=input_ids.device)
    torch.manual_seed(0)
    for _ in range(2):
        u = torch.randn(d, device=input_ids.device)
        acts = {}

        def rec(l, root):
            def f(m, i, o):
                h = o if torch.is_tensor(o) else o[0]
                if root:
                    h.requires_grad_(True)
                acts[l] = h

            return f

        with torch.enable_grad(), utils.hooks(model, {src: rec(src, True), tgt: rec(tgt, False)}):
            utils.get_text_model(model)(input_ids=input_ids, use_cache=False)
            scalar = (acts[tgt][0, valid].float() @ u).sum()
            (g,) = torch.autograd.grad(scalar, acts[src])
        expected = g[0, valid].float().mean(0)  # u^T J
        got = u @ J[src]
        cos = torch.nn.functional.cosine_similarity(expected, got, dim=0).item()
        rel = ((expected - got).norm() / expected.norm()).item()
        assert cos > 0.99 and rel < 0.1, (
            f"u^T J doesn't match the direct VJP (cos={cos:.4f}, rel err={rel:.3f}). Check that you put the one-hot "
            "cotangent at every valid target position, and average (not sum) over valid source positions."
        )
    print("All tests in `test_jacobian_for_prompt` passed!")


def test_clamped_swap(make_clamped_swap_hooks: Callable, model, input_ids, unembed):
    from part35_j_space import utils

    d = unembed.W_U.shape[1]
    torch.manual_seed(0)
    layers = [10, 11, 12]
    vs = {l: torch.randn(d, device=input_ids.device) for l in layers}
    vt = {l: torch.randn(d, device=input_ids.device) for l in layers}
    clean = utils.get_resid(model, input_ids)
    fns = make_clamped_swap_hooks(model, input_ids, vs, vt, alpha=1.0)
    with utils.hooks(model, fns):
        patched = utils.get_resid(model, input_ids)
    for l in layers:
        V = torch.stack([vs[l], vt[l]], 1)
        c_clean = clean[l, 1:] @ torch.linalg.pinv(V).T
        c_patched = patched[l, 1:] @ torch.linalg.pinv(V).T
        torch.testing.assert_close(
            c_patched, c_clean.flip(-1), atol=0.05 * c_clean.abs().max().item() + 1e-2, rtol=0.05,
            msg=f"At layer {l}, the patched coordinates should equal the *clean* coordinates swapped (clamping)",
        )
    torch.testing.assert_close(patched[:, 0], clean[:, 0], msg="Position 0 (BOS) should not be patched")
    print("All tests in `test_clamped_swap` passed!")


def test_nn_omp(nn_omp: Callable):
    torch.manual_seed(0)
    d, n, k = 64, 500, 5
    D = torch.nn.functional.normalize(torch.randn(n, d), dim=-1)
    idx = torch.randperm(n)[:k]
    coefs = torch.rand(k) + 0.5
    x = coefs @ D[idx]
    sel, c = nn_omp(x, D, k)
    assert len(sel) <= k and (c >= -1e-6).all(), "Coefficients must be non-negative and at most k atoms selected"
    recon = c @ D[sel]
    assert (x - recon).norm() / x.norm() < 1e-3, f"Should recover an exactly k-sparse signal (rel err {(x - recon).norm() / x.norm():.4f})"
    assert set(sel.tolist()) == set(idx.tolist()), "Should select the true atoms"
    print("All tests in `test_nn_omp` passed!")


def test_project_out(project_out: Callable):
    torch.manual_seed(0)
    d = 32
    V = torch.randn(d, 3)
    V[:, 1] = V[:, 0] + 0.1 * V[:, 1]  # nearly collinear columns
    h = torch.randn(5, d)
    out = project_out(h, V)
    assert (out @ V).abs().max() < 1e-4, "Output should be orthogonal to every column of V"
    P = V @ torch.linalg.pinv(V)
    torch.testing.assert_close(out, h - h @ P.T, atol=1e-4, rtol=1e-4)
    print("All tests in `test_project_out` passed!")


def test_lrp_conservation(lrp_mlp_forward: Callable, mlp):
    """With the identity + half rules, the 'LRP Jacobian' J~(x) satisfies J~(x) x = f(x) (bias-free MLP)."""
    torch.manual_seed(0)
    d = mlp.gate_proj.in_features
    x = (torch.randn(1, d, device=mlp.gate_proj.weight.device) * 3).to(mlp.gate_proj.weight.dtype).float()
    mlp_f = mlp.float()
    try:
        with torch.enable_grad():
            x_ = x.clone().requires_grad_(True)
            y = lrp_mlp_forward(mlp_f, x_)
            torch.testing.assert_close(y, mlp_f(x), msg="The LRP forward pass must give identical outputs")
            u = torch.randn_like(y)
            (g,) = torch.autograd.grad((y * u).sum(), x_)
        lhs = (g * x).sum()  # u^T J~ x
        rhs = (u * y).sum()  # u^T f(x)
        assert torch.isclose(lhs, rhs, rtol=1e-3), f"Conservation fails: u^T J~ x = {lhs:.4f} but u^T f(x) = {rhs:.4f}"
    finally:
        mlp.to(torch.bfloat16)
    print("All tests in `test_lrp_conservation` passed!")
