# %%

print('chickens')
# %%
import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

chapter = "chapter0_fundamentals"
repo = "ARENA_materials"
branch = "main"

# Install dependencies
try:
    import jaxtyping
except:
    %pip install einops jaxtyping

# Get root directory, handling 3 different cases: (1) Colab, (2) notebook not in ARENA repo, (3) notebook in ARENA repo
root = (
    "/content"
    if IN_COLAB
    else "/root"
    if repo not in os.getcwd()
    else str(next(p for p in Path.cwd().parents if p.name == repo))
)

if Path(root).exists() and not Path(f"{root}/{chapter}").exists():
    if not IN_COLAB:
        !sudo apt-get install unzip
        %pip install jupyter ipython --upgrade

    if not os.path.exists(f"{root}/{chapter}"):
        !wget -P {root} https://github.com/ARENA-education/ARENA_materials/archive/refs/heads/{branch}.zip
        !unzip {root}/{branch}.zip '{repo}-{branch}/{chapter}/exercises/*' -d {root}
        !mv {root}/{repo}-{branch}/{chapter} {root}/{chapter}
        !rm {root}/{branch}.zip
        !rmdir {root}/{repo}-{branch}


if f"{root}/{chapter}/exercises" not in sys.path:
    sys.path.append(f"{root}/{chapter}/exercises")

os.chdir(f"{root}/{chapter}/exercises")
# %%
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

Arr = np.ndarray
grad_tracking_enabled = True

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part4_backprop"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))


import part4_backprop.tests as tests
from part4_backprop.utils import get_mnist, visualize
from plotly_utils import line
# %%
def log_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    """Backwards function for f(x) = log(x)
    f'(x) = 1/x
    g(f(x)) = g(log(x))
    d g(f(x)) /dx = g'(log(x)) d (log(x)) / dx


    grad_out: Gradient of some loss wrt out
    out: the output of np.log(x).
    x: the input of np.log.

    Return: gradient of the given loss wrt x
    """
    return grad_out / x


tests.test_log_back(log_back)
# %%

def unbroadcast(broadcasted: Arr, original: Arr) -> Arr:
    """
    Sum 'broadcasted' until it has the shape of 'original'.

    broadcasted: An array that was formerly of the same shape of 'original' and was expanded by
        broadcasting rules.
    """
    # YOUR CODE HERE: sum over `broadcasted` until it has the shape of `original`

    og_shape = original.shape
    b_shape = broadcasted.shape

    original_b = original.copy()

    added_dims = 0
    while len(original.shape) != len(broadcasted.shape):
        original = np.expand_dims(original,axis=0)
        added_dims += 1
    sum_axis = []
    for i in range(len(original.shape)):
        if original.shape[i] != broadcasted.shape[i]:
            sum_axis.append(i)
    broadcasted = broadcasted.sum(axis=tuple(sum_axis),keepdims=True)
    return broadcasted.reshape(og_shape)
    


tests.test_unbroadcast(unbroadcast)
# %%

def multiply_back0(grad_out: Arr, out: Arr, x: Arr, y: Arr | float) -> Arr:
    """Backwards function for x * y wrt argument 0 aka x."""
    if not isinstance(y, Arr):
        y = np.array(y)
    
    return unbroadcast(y*grad_out,x)




def multiply_back1(grad_out: Arr, out: Arr, x: Arr | float, y: Arr) -> Arr:
    """Backwards function for x * y wrt argument 1 aka y."""
    if not isinstance(x, Arr):
        x = np.array(x)

    return unbroadcast(x*grad_out,y)

tests.test_multiply_back(multiply_back0, multiply_back1)
tests.test_multiply_back_float(multiply_back0, multiply_back1)
# %%

def forward_and_back(a: Arr, b: Arr, c: Arr) -> tuple[Arr, Arr, Arr]:
    """
    Calculates the output of the computational graph above (g), then backpropagates the gradients
    and returns dg/da, dg/db, and dg/dc.
    g = log(f)
    dg/df = ...
    f = d * e
    dg/dd = dg/df * df/dd
    dg/de = dg/df * df/de
    dg/da = dg/dd * dd/da
    dg/db = dg/dd * dd/db
    dg/dc = dg/de * de/dc
    """
    d = a * b
    e = np.log(c)
    f = d * e
    g = np.log(f)
    final_grad_out = np.ones_like(g)

    # YOUR CODE HERE - use your backward functions to compute the gradients of g wrt a, b, and c
    dg_df = log_back(final_grad_out,g,f)
    dg_dd = multiply_back0(dg_df,f,d,e)
    dg_de = multiply_back1(dg_df,f,d,e)
    dg_dc = log_back(dg_de,e,c)
    dg_da = multiply_back0(dg_dd,d,a,b)
    dg_db = multiply_back1(dg_dd,d,a,b)
    return (dg_da, dg_db, dg_dc)


tests.test_forward_and_back(forward_and_back)
# %%

@dataclass(frozen=True)
class Recipe:
    """Extra information necessary to run backpropagation. You don't need to modify this."""

    func: Callable
    "The 'inner' NumPy function that does the actual forward computation."
    "Note, we call it 'inner' to distinguish it from the wrapper we'll create for it later on."

    args: tuple
    "The input arguments passed to func."
    "For instance, if func=np.sum then args would be a length-1 tuple with the tensor to be summed."

    kwargs: dict[str, Any]
    "Keyword arguments passed to func."
    "For instance, if func was np.sum then kwargs might contain 'dim' and 'keepdims'."

    parents: dict[int, "Tensor"]
    "Map from positional argument index to the Tensor at that position."
    "For passing gradients back along the computational graph."
# %%

class BackwardFuncLookup:
    def __init__(self):
        self.lookup = dict()

    def add_back_func(self, forward_fn: Callable, arg_position: int, back_fn: Callable):
        self.lookup[forward_fn] = {'arg_position' : arg_position, 'back_fn' : back_fn}

    def get_back_func(self, forward_fn: Callable, arg_position: int) -> Callable:
        dicty = self.lookup[forward_fn]
        if forward_fn == np.multiply:
            ...
        else:
            return dicty['back_fn']


BACK_FUNCS = BackwardFuncLookup()

BACK_FUNCS.add_back_func(np.log, 0, log_back)
BACK_FUNCS.add_back_func(np.multiply, 0, multiply_back0)
BACK_FUNCS.add_back_func(np.multiply, 1, multiply_back1)

assert BACK_FUNCS.get_back_func(np.log, 0) == log_back
assert BACK_FUNCS.get_back_func(np.multiply, 0) == multiply_back0
assert BACK_FUNCS.get_back_func(np.multiply, 1) == multiply_back1

print("Tests passed - BackwardFuncLookup class is working as expected!")
# %%
