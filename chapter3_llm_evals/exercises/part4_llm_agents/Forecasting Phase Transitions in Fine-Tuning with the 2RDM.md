# Forecasting Phase Transitions in Fine-Tuning with the 2RDM

Oct 28, 2026 · @Joep

## Summary

A one-week project to test whether the 2RDM (Hennick & Corlouer, 2026) can **forecast when emergent misalignment switches on** during fine-tuning, and whether that signal is specific to misalignment.

The 2RDM is a cheap observable built from per-sample losses on a small probe set. The paper shows it can warn of phase transitions, but its emergent-misalignment (EM) experiment is thin: one setup, a transition within 5–10 steps, and no seeds, controls or measured lead time.

This project adds what is missing: a slowed-down transition, proper controls, a measured lead time, and an attempt to turn the warning into a forecast of the onset step. The forecast borrows from structural engineering, where buckling loads are predicted by extrapolating stiffness loss (Southwell plot).

**Decision this doc supports:** is this worth one week of capstone time, and what result would make it worthwhile?

## Background

The 2RDM is the covariance matrix of per-sample losses on a fixed probe set, taken over small changes to the weights. It costs only forward passes on the probe set, so it scales to large models.

**Two ways to get the weight changes:**

- *Dynamical:* the actual training steps in a sliding window (e.g. the last 20 checkpoints).
- *Gaussian:* small random perturbations around one checkpoint. Used when there are too few steps for a window.

**Three readouts:**

- **Spectral heat capacity (SHC):** the variance of the matrix's eigenvalues. It rises when loss fluctuations concentrate in a few modes. Under critical slowing down it provably rises *before* a continuous (second-order) transition.
- **Participation ratio (PR):** roughly how many modes take part in the change.
- **Top eigenvector:** which probe samples move together, i.e. *what* is changing.

**What the paper showed.** Validated on deep linear networks, induction heads, grokking and EM. On EM (Qwen2.5-7B, rank-1 LoRA, bad medical advice), the variance increase was concentrated in alignment probes, not benign or medical ones.

**Where it is thin:**

- EM shift was small (\~20%) and happened in steps 5–10, so only the Gaussian version could be used.
- No seeds, narrow-only control, false-positive rate or lead time reported for EM.
- With a trailing window, the SHC *peak* falls about W/2 steps after the steepest point, and its height scales as W⁴. Any real warning is in the *onset* of the rise, which needs careful measurement.
- The authors call probe-set design the most pressing open problem: the method only sees transitions the probes can resolve.

## Goals and research questions

**Main goal:** turn the 2RDM from a warning light into a calibrated forecast of misalignment onset, and check that it fires for misalignment rather than for any learning.

1. **Specificity.** Does the signal rise in EM runs but stay flat in narrow-only and benign control runs? What is the false-positive rate?
2. **Lead time.** With the transition slowed to hundreds of steps, how many steps before behavioural onset does the signal start rising?
3. **Forecast.** Can early checkpoints predict the onset step? Engineering analogue: extrapolate stiffness loss to zero to predict buckling (Southwell plot).
4. **What is forming.** Does the top eigenvector concentrate on the alignment probes before onset?

**Why it matters for safety:** a cheap, specific, early training-time monitor would let you stop or redirect fine-tuning before misalignment appears, including in settings where you can't run behavioural evals every step.

**Why this fits me:** two of the paper's open problems are standard problems in multiscale modelling. Probe selection resembles choosing integration points in reduced-order models (hyper-reduction). Predicting a transition from early data resembles stability analysis of structures.

**Stretch goals** (only if the core runs smoothly):

- *Nonlinearity check:* compute the Gaussian 2RDM at two perturbation sizes; drift from the expected σ² scaling measures linearisation breaking down, which may warn earlier.
- *Smarter probes:* greedy probe selection versus random, at equal probe count.
- *New setting:* apply it to my existing GRPO obfuscation run, with monitor-relevant prompts as probes.

## Experiments

All experiments share one training setup; they differ only in what is computed from the stored logs.

### Shared setup

- **Model organism:** ARENA 4.1 / *Model Organisms for EM* setup. Smallest Qwen2.5 model that shows EM (try 0.5B, fall back to 7B), rank-1 LoRA, bad-medical-advice data.
- **Slow the transition down:** constant, lowered learning rate, no warmup, so onset spans hundreds of steps rather than 5–10.
- **Runs:** EM × 3–5 seeds; *narrow-only* control (same data plus a KL penalty to the base model on unrelated prompts, so misbehaviour stays narrow); *benign* control (good medical advice, same format).
- **Probe set (\~200–400 prompts):** alignment probes (EM first-plot questions in free-form, JSON and templated formats), boundary-ambiguous but harmless questions, in-domain medical questions, benign capability questions.
- **Per-sample loss:** each probe gets a fixed aligned and a fixed misaligned reference answer; record the loss of each. Their gap is a dense, cheap order parameter.

**During training:**

- Every step: save the LoRA adapter (tiny).
- Every 2–5 steps: forward pass on all probes, store the per-probe loss vectors.
- Every 10–20 steps: sample answers to the alignment probes and score them with an LLM judge. This is the ground-truth misalignment rate.
- Define the onset step **t\*** as the point of steepest rise in the judge-scored misalignment rate, cross-checked against the loss gap.

### The four core experiments

| Experiment | What is computed | Plot | How to read it |
| --- | --- | --- | --- |
| E1 Specificity | SHC, PR and top eigenvalue λ₁ over a sliding window (W = 20; also 10 and 40) | Signal vs step with judge misalignment overlaid; one panel per run type, seeds shaded | Rises only in EM runs, before t\* → specific early warning. Rises in all runs → a generic "learning" detector. Rises only at or after t\* → concurrent detection, no warning |
| E2 Lead time | Warning step = first step the SHC exceeds its early-training baseline by 3 standard deviations (threshold fixed on control runs) | Lead time (t\* minus warning step) per seed, for each window size | Consistent positive lead across seeds → usable monitor. Lead shrinking to zero as W shrinks → the "warning" was a window artifact |
| E3 Forecast | Using only data up to a fraction f of t\*, fit 1/λ₁ vs step and extrapolate to zero; separately fit log λ₁ vs step | Southwell-style plot of 1/λ₁ vs step; forecast error vs f | 1/λ₁ falls on a straight line → critical-slowing-down picture, onset is forecastable. log λ₁ straight → saddle-escape picture, forecast via growth rate. Neither → no simple forecast |
| E4 What is forming | How concentrated the top eigenvector is (inverse participation ratio), and its weight on each probe group | Eigenvector weight per probe group vs step | Weight shifts onto alignment probes before t\* → the signal identifies *what* is forming, not just *that* something is |

**Compute:** forward passes on a few hundred probes every few steps are cheap. The judge sampling is the main cost and runs only every 10–20 steps.

## Limitations and risks

The biggest risk is Day 1: EM may not reproduce cleanly at small scale with a slowed-down transition.

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| EM too weak or absent at 0.5B, rank-1 | No transition, nothing to detect | Check first; fall back to 7B and the Gaussian 2RDM |
| Lowering the learning rate changes the transition itself | Results may not carry over to normal training | Run one seed at the normal rate as a reference |
| Loss on fixed reference answers may miss misbehaviour | The model can drift in ways these targets don't capture | Use both aligned and misaligned references; validate the gap against judge scores |
| Probe set chosen knowing the answer | Inflates specificity | Add a random probe subset and report both |
| Window size trades lag against noise | Peak lags about W/2; height scales as W⁴ | Report onset of the rise, not the peak; sweep W |
| Optimizer and schedule effects show up too | Warmup or Adam state can produce spikes unrelated to behaviour | Constant learning rate, no warmup; controls catch the rest |
| Few seeds, chosen threshold | Lead time could be noise | Fix the threshold on control runs before looking at EM runs |
| EM may be an abrupt (first-order) transition | Theory then predicts no early warning | Still a clear, publishable result: it says which kind of monitor can work |

**Scope risk:** the stretch goals are tempting. They come after E1–E4 are done, not alongside.

## Plan, deliverables and decision criteria

Five working days plus two buffer days, with a gate at the end of Day 1.

1. **Day 1:** reproduce EM with the slowed-down transition; build the probe set and logging. *Gate:* onset spans at least \~50 steps and is consistent across 2 seeds. If not, switch to 7B with the Gaussian 2RDM.
2. **Day 2:** run all seeds and both controls; compute SHC, PR and λ₁ (E1).
3. **Day 3:** lead time with a threshold fixed on controls (E2); window-size sweep.
4. **Day 4:** forecast fits (E3) and eigenvector analysis (E4).
5. **Day 5:** write-up and plots.
6. **Days 6–7:** buffer, or one stretch goal.

**Deliverables:** a public repo, four figures (one per experiment), and a short write-up or blog post.

**Worth doing if:**

- It answers a question the paper leaves open (specificity, lead time, forecastability) with controls the paper lacks.
- Every outcome is informative: a forecast works, the signal is generic, or EM gives no warning at all.
- It's feasible in a week on available compute, with a clear fallback at the Day 1 gate.

**Not worth doing if:**

- The authors or others have already published seeded, controlled EM results (check before starting).
- EM can't be made to unfold over enough steps at affordable scale.

**Open questions for discussion:**

- Is the reference-answer loss a good enough per-sample signal, or should probes use the judge score directly?
- Is "forecast the onset step" the right headline, or is "specific vs generic signal" more valuable?
- Should the GRPO obfuscation run replace EM as the main setting, since it's untested territory?

## Sources

- [Hennick & Corlouer (2026), From Density Matrices to Phase Transitions in Deep Learning](https://arxiv.org/html/2603.29805v2)
- [Turner, Soligo et al. (2025), Model Organisms for Emergent Misalignment](https://arxiv.org/html/2506.11613v1)
- [Arnold & Lörch (2025), Order Parameters for Emergent Misalignment](https://arxiv.org/html/2508.20015v1)
