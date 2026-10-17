---
name: hint
description: Syntax hint for an ARENA exercise - names every function, method, argument and tensor shape the exercise needs and explains what is wrong in the user's attempt, without writing the solution. Use when the user types /hint (optionally with an exercise or function name) or asks for a hint on an ARENA exercise.
---

# /hint: the syntax you need for an ARENA exercise, never the solution

The user is working through ARENA (repo with `chapter*_*/exercises/part*/`). They fill in exercises in their own file in the
exercise folder (`answers_c.py`, `answers.py`, or the exercises notebook). The goal is to unblock syntax: they often don't
know the *name* of the function an exercise implies, its arguments, or what shape it returns.

## 1. Find the exercise

- If the user named a function or exercise (e.g. `/hint compute_response_kl`), use it.
- Otherwise take the first function in their most recently modified `answers*.py` (current folder) whose body still has
  `raise NotImplementedError`, a bare `pass`, or `YOUR CODE HERE`; or the function named in an error they pasted.
- Read: their current attempt; the reference in `solutions.py` in the same folder (`def <name>`); the exercise text in
  `../../instructions/pages/*.md` (search for the exercise heading); and any test for it in `tests.py`.

## 2. Answer, tersely, in this order

1. **Functions you'll need** as a table: `call signature` | returns (type and shape) | what it's for. List every
   function, method, attribute and keyword argument the reference uses that someone might not know the name of, including
   easy-to-miss kwargs (`keepdim=True`, `padding_side="left"`, `return_tensors="pt"`, `log_target=True`) and context
   managers (`with model.disable_adapter():`). Skip Python basics unless they're misusing one.
2. **Shapes** (tensor exercises only): the shape at each stage with named dims (batch, seq, d_model, vocab, ...).
3. **Gotchas**: 1-3 traps specific to this exercise (argument order, off-by-one slices, dtype/device, in-place edits,
   hook registration order, string vs numeric sort...).
4. **Your attempt** (if they have one or pasted an error): which line is wrong and why, in words, and which function or
   argument fixes it.

## Rules

- Never write the solution, its body, or any line that could be pasted in as the answer. The only code allowed is a
  single call signature per function. Don't order the steps into a recipe that *is* the solution.
- Don't quote `solutions.py`. Don't reveal numbers the exercise asks them to find.
- If they explicitly ask for the solution, say where it is (`solutions.py`, the solutions notebook) instead.
- Signatures must match the installed versions (nodes can run newer libraries than the notebooks were written for):
  when unsure, check with `python -c "import inspect, torch.nn.functional as F; print(inspect.signature(F.kl_div))"`.
