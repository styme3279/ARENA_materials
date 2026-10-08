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

MAIN = __name__ == "__main__"

import part4_backprop.tests as tests
from part4_backprop.utils import get_mnist, visualize
from plotly_utils import line

# %%
def log_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    """Backwards function for f(x) = log(x)

    grad_out: Gradient of some loss wrt out
    out: the output of np.log(x).
    x: the input of np.log.

    Return: gradient of the given loss wrt x
    """
    return grad_out / x

tests.test_log_back(log_back)

# %%
x = np.ones((3, 1, 5))
y = np.ones((1, 4, 5))

z = x + y

# %%
x = np.ones((8, 2, 6))
y = np.ones((8, 2))

z = x + y

# %%
x = np.ones((4, 1))
y = np.ones((4,))

z = x + y

# %%
def unbroadcast(broadcasted: Arr, original: Arr) -> Arr:
    """
    Sum 'broadcasted' until it has the shape of 'original'.

    broadcasted: An array that was formerly of the same shape of 'original' and was expanded by
        broadcasting rules.
    """

    print(broadcasted.shape, original.shape)
    n_dim_broadcasted = len(broadcasted.shape)
    print(n_dim_broadcasted)

    n_dim_original = len(original.shape)
    print(n_dim_original)

    assert n_dim_broadcasted >= n_dim_original
    n_prepanded_dims = n_dim_broadcasted - n_dim_original
    for i in range(n_prepanded_dims):
        broadcasted = broadcasted.sum(axis=0)
        print(broadcasted.shape)

    assert n_dim_broadcasted == n_dim_original
    for i, o in enumerate(original.shape):
        if len(o) != len(broadcasted.shape[i]):
            broadcasted = broadcasted.sum(axis=i, keepdims=True)

    assert broadcasted.shape == original.shape
    return broadcasted

tests.test_unbroadcast(unbroadcast)

# %%
