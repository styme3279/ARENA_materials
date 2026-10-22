import copy
import dataclasses
import gc

import torch as t
import torch.nn as nn
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from part4_rlhf.tests import (
    _check,
    _check_rollout,
    _make_reward_fn,
    _record_grad_modes,
    _solutions,
    _tiny_model_dir,
    _with_forked_rng,
)

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")


def _is_lora(module: nn.Module) -> bool:
    return hasattr(module, "A") and hasattr(module, "B") and hasattr(module, "base")


def _reference_lora(solutions, lora: nn.Module, base: nn.Linear, rank: int, alpha: float) -> nn.Module:
    """The solution's LoRALinear, wrapping a copy of `base`, with the same A and B as `lora`."""
    ref = solutions.LoRALinear(copy.deepcopy(base), rank=rank, alpha=alpha)
    with t.no_grad():
        ref.A.copy_(lora.A)
        ref.B.copy_(lora.B)
    return ref


def _check_lora_forward(lora: nn.Module, ref: nn.Module, x: t.Tensor, name: str):
    """Checks the output of `lora` on `x`, and its gradients wrt x, A and B, against the solution's `ref`."""
    x_copy = x.clone()
    x, x_ref = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    lora.zero_grad(set_to_none=True)
    ref.zero_grad(set_to_none=True)
    out = lora(x)
    assert t.equal(x.detach(), x_copy), "forward shouldn't modify its input in place"
    expected = ref(x_ref)
    update = expected - ref.base(x_ref)
    _check(
        out,
        expected,
        [
            (ref.base(x_ref) + update / ref.scaling, "Looks like you forgot to scale the update by alpha / rank."),
            (
                ref.base(x_ref) + update / ref.scaling**2,
                "Looks like you scaled the update by rank / alpha. It should be alpha / rank.",
            ),
            (ref.base(x_ref), "The output is just the base layer's: did you forget to add the low-rank update?"),
        ],
        name,
        atol=1e-5,
        rtol=1e-4,
    )
    weights = t.randn(out.shape, generator=t.Generator().manual_seed(0))
    (out * weights).sum().backward()
    (expected * weights).sum().backward()
    assert lora.A.grad is not None and lora.B.grad is not None, "No gradients reached A and B"
    assert all(p.grad is None for p in lora.base.parameters()), "The base layer is frozen, so it shouldn't get gradients"
    for grad, ref_grad, what in [(lora.A.grad, ref.A.grad, "A"), (lora.B.grad, ref.B.grad, "B"), (x.grad, x_ref.grad, "x")]:
        message = f"{name}: wrong gradient wrt {what}."
        t.testing.assert_close(grad, ref_grad, atol=1e-5, rtol=1e-4, msg=lambda m: f"{message}\n{m}")


@_with_forked_rng
def test_lora_linear(LoRALinear):
    solutions = _solutions()
    t.manual_seed(0)
    d_in, d_out, rank, alpha = 512, 48, 4, 8
    base = nn.Linear(d_in, d_out, bias=True)
    base_copy = copy.deepcopy(base)
    lora = LoRALinear(base, rank=rank, alpha=alpha)

    # Shapes, dtypes and trainability
    assert lora.A.shape == (d_in, rank), f"A should have shape (in_features, rank) = {(d_in, rank)}, got {tuple(lora.A.shape)}"
    assert lora.B.shape == (rank, d_out), f"B should have shape (rank, out_features) = {(rank, d_out)}, got {tuple(lora.B.shape)}"
    assert isinstance(lora.A, nn.Parameter) and isinstance(lora.B, nn.Parameter), "A and B should be nn.Parameters"
    assert lora.A.dtype == lora.B.dtype == t.float32, "A and B should be float32"
    assert lora.A.requires_grad and lora.B.requires_grad, "A and B should be trainable"
    assert not any(p.requires_grad for p in base.parameters()), "The base layer's parameters should be frozen"
    assert lora.base is base, "self.base should be the base layer"
    assert all(t.equal(p, q) for p, q in zip(base.parameters(), base_copy.parameters())), "The base layer shouldn't change"
    assert {id(p) for p in lora.parameters() if p.requires_grad} == {id(lora.A), id(lora.B)}, (
        "A and B should be the only trainable parameters"
    )
    assert abs(lora.scaling - alpha / rank) < 1e-8, f"scaling should be alpha / rank = {alpha / rank}, got {lora.scaling}"

    # Initialization: B is zero, A is Gaussian with mean 0 and std 1/sqrt(in_features). A has d_in * rank = 2048
    # entries, so the sample mean and std are accurate to about 2% of the std, well within the tolerances below. And
    # d_in != d_out != rank, so using the wrong one in the std shows up clearly.
    assert (lora.B == 0).all(), "B should be initialized to zeros"
    assert (lora.A != 0).any(), "A should be initialized randomly, not to zeros (otherwise B never gets a gradient)"
    sigma = d_in**-0.5
    mean, std = lora.A.mean().item(), lora.A.std().item()
    for wrong, name in [(d_out**-0.5, "1/sqrt(out_features)"), (rank**-0.5, "1/sqrt(rank)"), (1 / rank, "1/rank"), (1 / d_in, "1/in_features"), (1.0, "1")]:
        if abs(std - wrong) < 0.1 * wrong:
            raise Exception(f"A has std {std:.4f}, which looks like {name}. It should be 1/sqrt(in_features) = {sigma:.4f}.")
    assert abs(std - sigma) < 0.1 * sigma, f"A should have std 1/sqrt(in_features) = {sigma:.4f}, got {std:.4f}"
    assert abs(mean) < 0.1 * std, f"A should have mean 0, got {mean:.4f} (with std {std:.4f})"
    # A uniform distribution with this std only reaches sqrt(3) = 1.73 std devs, while the most extreme of 2048 Gaussian
    # samples is beyond 2 std devs with probability 1 - (0.977)^2048, i.e. essentially always
    assert lora.A.abs().max() > 2 * sigma, "A should be Gaussian (no values beyond 2 std devs: is it uniform?)"
    # Two adapters shouldn't get the same A (e.g. from reseeding inside __init__)
    assert not t.equal(LoRALinear(nn.Linear(d_in, d_out), rank=rank, alpha=alpha).A, lora.A), "A should be random"

    # At initialization, the output equals the base layer's, and B already gets a (nonzero) gradient: the update
    # x A B has to be computed even though it's zero, or B never starts training
    x = t.randn(2, 3, d_in)
    out = lora(x)
    t.testing.assert_close(out, base(x))
    (out * t.randn(out.shape)).sum().backward()
    assert lora.B.grad is not None and (lora.B.grad != 0).any(), (
        "At initialization (B = 0), B should still get a nonzero gradient: don't skip the update when B is zero"
    )
    lora.zero_grad(set_to_none=True)

    # With a nonzero B: h = x W_0 + b + (alpha / rank) x A B. We compare the output and the gradients (wrt x, A and B)
    # to the solution's, with inputs with several batch dims, or none, and a base layer with and without a bias
    with t.no_grad():
        lora.B.normal_()
    ref = _reference_lora(solutions, lora, base, rank, alpha)
    _check_lora_forward(lora, ref, x, "The output of LoRALinear")
    _check_lora_forward(lora, ref, t.randn(d_in), "The output of LoRALinear (on an input with no batch dims)")
    d_in_2, d_out_2, rank_2, alpha_2 = 24, 40, 3, 5  # a non-integer alpha / rank, and d_out > d_in
    base_2 = nn.Linear(d_in_2, d_out_2, bias=False)
    lora_2 = LoRALinear(base_2, rank=rank_2, alpha=alpha_2)
    assert abs(lora_2.scaling - alpha_2 / rank_2) < 1e-8, f"scaling should be alpha / rank, got {lora_2.scaling}"
    with t.no_grad():
        lora_2.B.normal_()
    ref_2 = _reference_lora(solutions, lora_2, base_2, rank_2, alpha_2)
    _check_lora_forward(lora_2, ref_2, t.randn(5, d_in_2), "The output of LoRALinear (base layer without a bias)")

    # Disabled: acts exactly like the base layer (including gradients flowing back to the input)
    lora.enabled = False
    x_grad = x.clone().requires_grad_(True)
    out = lora(x_grad)
    t.testing.assert_close(out, base(x), msg="With self.enabled = False, forward should skip the low-rank update")
    assert out.requires_grad, "With self.enabled = False, gradients should still flow back to the input"
    out.sum().backward()
    t.testing.assert_close(x_grad.grad, base.weight.sum(0).expand_as(x), msg="Wrong gradient wrt x when disabled")
    lora.zero_grad(set_to_none=True)
    lora.enabled = True
    _check_lora_forward(lora, ref, x, "The output of LoRALinear (after switching it off and on again)")

    # A bfloat16 base layer: the adapters stay float32, and the output is bfloat16
    base_bf16 = nn.Linear(d_in, d_out, dtype=t.bfloat16)
    lora_bf16 = LoRALinear(base_bf16, rank=rank, alpha=alpha)
    assert lora_bf16.A.dtype == lora_bf16.B.dtype == t.float32, "A and B should be float32 even if the base is bfloat16"
    with t.no_grad():
        lora_bf16.B.normal_()
    x = t.randn(5, d_in, dtype=t.bfloat16)
    try:
        out = lora_bf16(x)
    except RuntimeError as e:
        raise Exception(
            f"forward failed on a bfloat16 input ({e}). Cast x to A's dtype (float32) before multiplying by A and B, and "
            "cast the update back to the output's dtype at the end."
        ) from e
    assert out.dtype == t.bfloat16, f"Output should have the input's dtype (bfloat16), got {out.dtype}"
    expected = base_bf16(x).float() + (alpha / rank) * (x.float() @ lora_bf16.A @ lora_bf16.B)
    t.testing.assert_close(out.float(), expected, atol=0.1, rtol=0.02)

    # The adapters should be created on the base layer's device
    if device.type != "cpu":
        lora_dev = LoRALinear(nn.Linear(d_in, d_out).to(device), rank=rank, alpha=alpha)
        assert lora_dev.A.device.type == lora_dev.B.device.type == device.type, (
            f"A and B should be on the base layer's device ({device}), got {lora_dev.A.device} and {lora_dev.B.device}"
        )

    print("All tests in `test_lora_linear` passed!")


@_with_forked_rng
def test_lora_merge(LoRALinear):
    solutions = _solutions()
    t.manual_seed(0)
    d_in, d_out, rank, alpha = 64, 32, 4, 16
    base = nn.Linear(d_in, d_out)
    weight_before, bias_before = base.weight.detach().clone(), base.bias.detach().clone()
    lora = LoRALinear(base, rank=rank, alpha=alpha)
    with t.no_grad():
        lora.B.normal_()
    A, B = lora.A.detach().clone(), lora.B.detach().clone()
    ref = _reference_lora(solutions, lora, base, rank, alpha)
    x = t.randn(10, d_in)
    out_before = lora(x).detach()

    try:
        lora.merge()
    except RuntimeError as e:
        raise Exception(
            f"merge failed ({e}). Remember that base.weight has shape (out_features, in_features), the transpose of A B."
        ) from e
    ref.merge()
    assert lora.merged, "merge should set self.merged = True"
    assert t.equal(lora.A, A) and t.equal(lora.B, B), "merge shouldn't change A or B"
    assert t.equal(base.bias, bias_before), "merge shouldn't change the bias"
    assert isinstance(base.weight, nn.Parameter) and not base.weight.requires_grad, "base.weight should stay frozen"
    # nn.Linear stores its weight as (out_features, in_features), i.e. the transpose of our W_0
    _check(
        base.weight,
        ref.base.weight,
        [(weight_before + (A @ B).T, "Looks like you forgot to scale the update by alpha / rank when merging.")],
        "base.weight after merging (it should be (W_0 + (alpha / rank) A B).T, since nn.Linear stores its weight "
        "transposed)",
    )
    out_after = lora(x)
    if t.allclose(out_after - out_before, (x @ A @ B) * lora.scaling, atol=1e-4):
        raise Exception("The update is added twice after merging: forward shouldn't add it once self.merged is True.")
    t.testing.assert_close(out_after, out_before, atol=1e-4, rtol=1e-4)
    x_grad = x.clone().requires_grad_(True)
    lora(x_grad).sum().backward()
    t.testing.assert_close(
        x_grad.grad, base.weight.sum(0).expand_as(x), msg="After merging, gradients should flow back to the input"
    )

    # unmerge should subtract the same update back out (exact up to float32 rounding here), and switch the adapter back on
    lora.unmerge()
    assert not lora.merged, "unmerge should set self.merged = False"
    assert t.equal(lora.A, A) and t.equal(lora.B, B), "unmerge shouldn't change A or B"
    assert t.equal(base.bias, bias_before), "unmerge shouldn't change the bias"
    t.testing.assert_close(
        base.weight,
        weight_before,
        atol=1e-5,
        rtol=1e-5,
        msg="After unmerge, base.weight should be back to W_0: subtract the same (alpha / rank) (A B).T that merge added",
    )
    out_unmerged = lora(x)
    if t.allclose(out_unmerged, base(x), atol=1e-4):
        raise Exception("After unmerge, forward should add the adapter's update again: did you set self.merged = False?")
    t.testing.assert_close(out_unmerged, out_before, atol=1e-4, rtol=1e-4)

    # A bfloat16 base layer: merging should keep base.weight in bfloat16 (and in place), adding the float32 update
    # with a single rounding at the end (e.g. `self.base.weight += delta`), rather than rounding the update to bfloat16
    # first
    base_bf16 = nn.Linear(d_in, d_out, dtype=t.bfloat16)
    weight_bf16 = base_bf16.weight.detach().clone()
    lora_bf16 = LoRALinear(base_bf16, rank=rank, alpha=alpha)
    with t.no_grad():
        lora_bf16.B.normal_()
    lora_bf16.merge()
    assert base_bf16.weight.dtype == t.bfloat16, (
        f"merge changed base.weight's dtype to {base_bf16.weight.dtype}: update it in place (e.g. with +=), which "
        "keeps the base layer's dtype"
    )
    delta = (lora_bf16.scaling * (lora_bf16.A @ lora_bf16.B)).T.detach()
    assert t.equal(base_bf16.weight, (weight_bf16.float() + delta).bfloat16()), (
        "With a bfloat16 base layer, merge should add the float32 update to the weights and round once (e.g. "
        "`self.base.weight += delta`), not round the update to bfloat16 first"
    )
    out = lora_bf16(t.randn(3, d_in, dtype=t.bfloat16))
    assert out.dtype == t.bfloat16, f"After merging, the output should still be bfloat16, got {out.dtype}"
    lora_bf16.unmerge()
    assert base_bf16.weight.dtype == t.bfloat16, f"unmerge changed base.weight's dtype to {base_bf16.weight.dtype}"
    # (merging and unmerging each round to bfloat16, which has about 3 significant figures)
    scale = (weight_bf16.float().abs() + delta.abs()).max()
    assert (base_bf16.weight.float() - weight_bf16.float()).abs().max() <= 0.01 * scale, (
        "After unmerge, base.weight should be back to (approximately, after bfloat16 rounding) W_0"
    )

    print("All tests in `test_lora_merge` passed!")


def _tiny_llama() -> LlamaForCausalLM:
    t.manual_seed(0)
    config = LlamaConfig(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(config).eval()


class _ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 6)
        self.k_q_proj = nn.Linear(8, 6)  # contains "q_proj", but isn't named "q_proj"
        self.q_proj_2 = nn.Linear(8, 6)  # likewise
        self.v_proj = nn.Sequential(nn.Linear(8, 6))  # named "v_proj", but not an nn.Linear


@_with_forked_rng
def test_apply_lora(apply_lora):
    model = _tiny_llama()
    tokens = t.randint(0, 100, (2, 7))
    logits_before = model(tokens).logits
    weights_before = {name: p.detach().clone() for name, p in model.named_parameters()}

    lora_modules = apply_lora(model, target_modules=("q_proj", "v_proj"), rank=4, alpha=4)
    assert len(lora_modules) == 4, f"Expected 4 LoRA modules (q_proj and v_proj in 2 layers), got {len(lora_modules)}"

    for layer in model.model.layers:
        attn = layer.self_attn
        assert _is_lora(attn.q_proj) and _is_lora(attn.v_proj), "q_proj and v_proj should be wrapped in LoRALinear"
        assert type(attn.k_proj) is nn.Linear and type(attn.o_proj) is nn.Linear, "k_proj and o_proj should be untouched"
        assert type(layer.mlp.up_proj) is nn.Linear, "The MLP layers should be untouched"
        assert type(attn.q_proj.base) is nn.Linear, "Each LoRALinear should wrap the original nn.Linear"
    assert set(map(id, lora_modules)) == {id(m) for m in model.modules() if _is_lora(m)}, (
        "apply_lora should return the LoRALinear modules it added to the model"
    )

    # The wrapped layers keep their weights
    for name, p in model.named_parameters():
        if not name.endswith((".A", ".B")):
            assert t.equal(p, weights_before[name.replace(".base.", ".")]), f"apply_lora changed the weights of {name}"

    # The adapters are trainable, and the wrapped layers are frozen (by LoRALinear)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    adapters = {name for name, _ in model.named_parameters() if name.endswith((".A", ".B"))}
    wrapped = {name for name, _ in model.named_parameters() if ".base." in name}
    assert adapters <= trainable, f"The LoRA A and B matrices should be trainable: {sorted(adapters - trainable)[:5]}"
    assert not (wrapped & trainable), f"The wrapped layers should be frozen: {sorted(wrapped & trainable)[:5]}"

    # The model's output is unchanged at initialization, and the shapes respect grouped-query attention
    t.testing.assert_close(model(tokens).logits, logits_before)
    k_v_dim = model.config.num_key_value_heads * (model.config.hidden_size // model.config.num_attention_heads)
    assert model.model.layers[0].self_attn.v_proj.B.shape[1] == k_v_dim
    for m in lora_modules:
        assert m.A.shape[1] == 4 and m.scaling == 1.0, "apply_lora should pass its `rank` and `alpha` to LoRALinear"

    # Other target modules, rank and alpha (in case any of these were hardcoded)
    model = _tiny_llama()
    lora_modules = apply_lora(model, target_modules=("k_proj", "o_proj", "up_proj"), rank=2, alpha=6)
    assert len(lora_modules) == 6, f"Expected 6 LoRA modules (k_proj, o_proj and up_proj in 2 layers), got {len(lora_modules)}"
    layer = model.model.layers[0]
    assert all(_is_lora(m) for m in (layer.self_attn.k_proj, layer.self_attn.o_proj, layer.mlp.up_proj)), (
        "apply_lora should wrap exactly the modules named in `target_modules`"
    )
    assert not any(_is_lora(m) for m in (layer.self_attn.q_proj, layer.self_attn.v_proj, layer.mlp.down_proj)), (
        "apply_lora should only wrap the modules named in `target_modules`"
    )
    assert all(m.A.shape[1] == 2 and m.scaling == 3.0 for m in lora_modules), (
        "apply_lora should pass its `rank` and `alpha` to LoRALinear"
    )

    # A toy model, to check the name matching: only nn.Linear modules whose name (the last part of the path) is exactly
    # one of the targets get wrapped, wherever they are in the model (including at the top level)
    toy = nn.Module()
    toy.q_proj = nn.Linear(8, 8)
    toy.block = _ToyBlock()
    toy.blocks = nn.ModuleList([_ToyBlock()])
    lora_modules = apply_lora(toy, target_modules=("q_proj", "v_proj"), rank=2, alpha=2)
    assert all(_is_lora(m) for m in (toy.q_proj, toy.block.q_proj, toy.blocks[0].q_proj)), (
        "apply_lora should wrap every nn.Linear named in `target_modules`, at any depth (including the top level)"
    )
    for block in (toy.block, toy.blocks[0]):
        assert type(block.k_q_proj) is nn.Linear and type(block.q_proj_2) is nn.Linear, (
            "apply_lora should only wrap modules whose name is exactly one of `target_modules` (e.g. not `k_q_proj`)"
        )
        assert type(block.v_proj) is nn.Sequential and type(block.v_proj[0]) is nn.Linear, (
            "apply_lora should only wrap nn.Linear modules"
        )
    assert len(lora_modules) == 3, f"Expected 3 LoRA modules in the toy model, got {len(lora_modules)}"

    print("All tests in `test_apply_lora` passed!")


@_with_forked_rng
@t.no_grad()
def test_lora_disabled(model, lora_disabled, tokenizer):
    device = next(model.parameters()).device
    tokens = tokenizer("The quick brown fox jumps over the lazy dog", return_tensors="pt").input_ids.to(device)
    lora_modules = [m for m in model.modules() if _is_lora(m)]
    assert len(lora_modules) > 0, "The model has no LoRA modules"

    # At initialization (B = 0), the adapted model is exactly the base model
    logits_init = model(tokens).logits
    with lora_disabled(model):
        t.testing.assert_close(model(tokens).logits, logits_init)

    # With nonzero B, the adapters change the output, but switching them off recovers the base model
    for m in lora_modules:
        m.B.normal_(std=0.01)
    logits_adapted = model(tokens).logits
    assert not t.allclose(logits_adapted, logits_init), "Changing B should change the model's output"
    with lora_disabled(model):
        t.testing.assert_close(model(tokens).logits, logits_init)
    t.testing.assert_close(model(tokens).logits, logits_adapted, msg="The adapters should be enabled again afterwards")

    # The adapters should be switched back on even if there's an error inside the `with` block
    class _Interrupt(Exception):
        pass

    try:
        with lora_disabled(model):
            raise _Interrupt
    except _Interrupt:
        pass
    t.testing.assert_close(
        model(tokens).logits, logits_adapted, msg="After an error inside `with lora_disabled(model)`, the adapters are off"
    )

    for m in lora_modules:
        m.B.zero_()
    print("All tests in `test_lora_disabled` passed!")


@_with_forked_rng
def test_grpo_trainer_lora(GRPO_LoRA_Trainer, GRPO_LoRA_Args, LoRALinear):
    """Tests GRPO_LoRA_Trainer on a tiny randomly-initialized Llama-style model (like SmolLM or Qwen)."""
    precision = t.get_float32_matmul_precision()
    t.set_float32_matmul_precision("highest")
    try:
        with _tiny_model_dir("llama") as model_dir:
            _test_grpo_trainer_lora(GRPO_LoRA_Trainer, GRPO_LoRA_Args, LoRALinear, model_dir)
    finally:
        t.set_float32_matmul_precision(precision)
    gc.collect()
    if t.cuda.is_available():
        t.cuda.empty_cache()
    print("All tests in `test_grpo_trainer_lora` passed!")


def _test_grpo_trainer_lora(GRPO_LoRA_Trainer, GRPO_LoRA_Args, LoRALinear, model_dir):
    solutions = _solutions()
    g = t.Generator().manual_seed(0)
    trainer = None
    reward_fn = _make_reward_fn(lambda: next(trainer.model.parameters()).device)
    args = GRPO_LoRA_Args(
        base_model=model_dir,
        prompt="One two three",
        batch_size=8,
        num_minibatches=2,
        gen_len=4,
        total_phases=2,
        warmup_steps=1,  # so the LR schedule isn't flat
        lr=1e-2,
        reward_fn=reward_fn,
        lora_rank=3,
        lora_alpha=12,
        target_modules=("q_proj", "k_proj", "up_proj"),
    )
    try:
        trainer = GRPO_LoRA_Trainer(args)
    except ValueError as e:
        if "empty parameter list" in str(e):
            raise Exception(
                "The optimizer has no parameters to train: did you freeze the model *after* applying LoRA? That freezes "
                "the adapters too. Freeze the base model first, then apply LoRA."
            ) from e
        raise
    model = trainer.model
    device = next(model.parameters()).device
    gen_len = args.gen_len

    assert trainer.ref_model is None or trainer.ref_model is model, (
        "Don't load a separate reference model: use the LoRA model with adapters disabled"
    )
    lora_modules = [m for m in model.modules() if isinstance(m, LoRALinear)]
    assert len(lora_modules) == 3 * model.config.num_hidden_layers, (
        f"Expected LoRA on args.target_modules = {args.target_modules} in every layer"
    )
    attn = model.model.layers[0].self_attn
    assert _is_lora(attn.k_proj) and _is_lora(model.model.layers[0].mlp.up_proj) and not _is_lora(attn.v_proj), (
        "Use args.target_modules, not the default targets"
    )
    assert all(m.A.shape[1] == args.lora_rank and m.scaling == args.lora_alpha / args.lora_rank for m in lora_modules), (
        "Pass args.lora_rank and args.lora_alpha to apply_lora"
    )
    assert next(model.parameters()).dtype == args.dtype, f"The base model should be loaded in {args.dtype}"
    lora_params = {id(p) for m in lora_modules for p in (m.A, m.B)}
    optimized = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    assert optimized == lora_params, (
        "The optimizer should train exactly the LoRA A and B matrices: did you freeze the base model before applying LoRA?"
    )
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert trainable == lora_params, "Only the LoRA A and B matrices should be trainable: freeze the base model"

    # The optimizer and LR schedule should be the ones from get_optimizer_and_scheduler: AdamW doing gradient *ascent*
    # (the inherited learning_phase calls backward on the objective), with lr = args.lr and the warmup + decay schedule
    ref_optimizer, ref_scheduler = solutions.get_optimizer_and_scheduler(args, nn.Linear(1, 1))
    for group in trainer.optimizer.param_groups:
        assert group.get("maximize", False), (
            "The optimizer should have maximize=True (we maximize the objective): use get_optimizer_and_scheduler"
        )
        assert group.get("initial_lr", group["lr"]) == args.lr and group["lr"] == ref_optimizer.param_groups[0]["lr"], (
            f"The optimizer's learning rate should start at args.lr * (warmup factor) = "
            f"{ref_optimizer.param_groups[0]['lr']:.3g}, got {group['lr']:.3g}: use get_optimizer_and_scheduler"
        )
    expected_lrs = []
    for _ in range(args.total_phases):
        ref_optimizer.step()
        ref_scheduler.step()
        expected_lrs.append(ref_optimizer.param_groups[0]["lr"])

    # The base model, loaded separately, to compare against
    base_model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=args.dtype).to(device).eval()
    atol = 1e-4 if args.dtype == t.float32 else 2e-2

    @t.no_grad()
    def logprobs_under(m, sample_ids):
        return solutions.get_logprobs(solutions.get_logits(m, sample_ids, gen_len), sample_ids, gen_len)

    # The reference logprobs are the base model's, whatever the adapters are
    sample_ids = t.randint(0, model.config.vocab_size, (5, 9), generator=g).to(device)
    base_lp = logprobs_under(base_model, sample_ids)
    for perturb in [False, True]:
        if perturb:
            with t.no_grad():
                for m in lora_modules:
                    m.B.normal_(std=0.3)
        adapted_lp = logprobs_under(model, sample_ids)
        ref_lp = trainer.get_ref_logprobs(sample_ids)
        assert ref_lp.shape == (5, gen_len), f"Expected shape {(5, gen_len)}, got {tuple(ref_lp.shape)}"
        assert not ref_lp.requires_grad, "get_ref_logprobs should run without gradients"
        t.testing.assert_close(
            ref_lp.float(), base_lp.float(), atol=atol, rtol=0, msg="get_ref_logprobs should switch the adapters off"
        )
        assert all(m.enabled for m in lora_modules), "After get_ref_logprobs, the adapters should be switched back on"
        t.testing.assert_close(logprobs_under(model, sample_ids), adapted_lp, msg="get_ref_logprobs changed the model")

    # The rollout phase (inherited from GRPO_Trainer) should use the adapted model for the policy's logprobs, and the
    # model with the adapters switched off for the reference logprobs
    trainer.step, trainer.phase = 0, 0
    grad_modes, handles = _record_grad_modes(model)
    try:
        memory = trainer.rollout_phase()
    except (AttributeError, TypeError) as e:
        if "NoneType" not in str(e):
            raise
        raise Exception(
            f"rollout_phase failed ({e!r}). Did it use self.ref_model directly? Use self.get_ref_logprobs, so that "
            "GRPO_LoRA_Trainer can override how the reference logprobs are computed."
        ) from e
    finally:
        for h in handles:
            h.remove()
    with t.no_grad():
        policy_lp, ref_lp = logprobs_under(model, memory.sample_ids), logprobs_under(base_model, memory.sample_ids)
    _check_rollout(trainer, memory, policy_lp, ref_lp, grad_modes, atol=atol)

    with t.no_grad():
        for m in lora_modules:
            m.B.zero_()
    del base_model

    # A short training run, which should only update the adapters
    base_weights = {name: p.detach().clone() for name, p in model.named_parameters() if not name.endswith((".A", ".B"))}
    adapters = {name: p.detach().clone() for name, p in model.named_parameters() if name.endswith((".A", ".B"))}
    trainer.train()
    lr = trainer.optimizer.param_groups[0]["lr"]
    assert abs(lr - expected_lrs[-1]) < 1e-12, (
        f"After training, the learning rate should be {expected_lrs[-1]:.3g} (from the LR schedule), got {lr:.3g}"
    )
    for name, p in model.named_parameters():
        if name in base_weights:
            assert t.equal(p, base_weights[name]), f"Training changed the base model's weights ({name})"
        else:
            assert not t.equal(p, adapters[name]), f"Training didn't update the LoRA matrix {name}"
    del trainer, model

    # The base model's dtype should come from args.dtype (we only check construction here, with the other dtype)
    other_dtype = t.float32 if args.dtype == t.bfloat16 else t.bfloat16
    args = dataclasses.replace(args, dtype=other_dtype)
    model = GRPO_LoRA_Trainer(args).model
    assert next(model.parameters()).dtype == other_dtype, f"The base model should be loaded in args.dtype = {other_dtype}"
    assert all(p.dtype == t.float32 for p in model.parameters() if p.requires_grad), "The adapters should be float32"
