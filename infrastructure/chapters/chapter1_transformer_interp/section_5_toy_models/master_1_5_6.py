# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
```python
[
    {"title": "The Maths of Parameter Decomposition", "icon": "1-circle-fill", "subtitle": "(25%)"},
    {"title": "SPD & VPD in Toy Models of Superposition", "icon": "2-circle-fill", "subtitle": "(40%)"},
    {"title": "Interpreting a Real LM's Parameters", "icon": "3-circle-fill", "subtitle": "(35%)"},
    {"title": "Bonus", "icon": "star", "subtitle": ""},
]
```
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# [1.5.5] Parameter Decomposition (SPD & VPD)
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# Introduction
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Most of the interpretability tools that exist and which we have looked at so far focuses on is **activation-based**. SAEs and Transcoders and NLAs and Activation Oracles and linear probes focus on the **activations** as the core thing to study. 

However, one could argue that to truely understand what a model does to the same level that we understand Grokking of Modular Addition networks, we must understand what all the parameters of the model are actually doing.

This set of material covers **linear parameter decomposition**, a family of methods developed in 2025-26 that attempts to break down model weights into more interpretable sub-components. (APD → SPD → VPD, by Bushnaq et. al). This decomposes a network's *weights* into a sum of parameter components.

Some benefits:

* The same method decomposes any linear layers, meaning both MLPs and attention layers are handled automatically. It also is able to find attention computations that are spread across multiple heads.
* The decomposition is resistant to feature splitting. 
* Because components live in parameter space, you can edit the model's algorithm by hand 

We will start by implementing [Stochastic Parameter Decomposition](https://arxiv.org/abs/2506.20790) (SPD), before moving on to understanding [adVersarial Parameter Decomposition](https://www.goodfire.ai/research/interpreting-lm-parameters) (VPD).

We first try the methods on a toy model of superposition (where you can skip some detail if you covered it a different day). Then we will load and investigate Goodfire's decomposition of their 4-layer 67M-parameter Pile model.
'''


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Setup (don't read, just run)
'''

# ! CELL TYPE: code
# ! FILTERS: [~]
# ! TAGS: []

from IPython import get_ipython

ipython = get_ipython()
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

# ! CELL TYPE: code
# ! FILTERS: [colab]
# ! TAGS: [master-comment]

# import os
# import sys
# from pathlib import Path
#
# IN_COLAB = "google.colab" in sys.modules
#
# chapter = "chapter1_transformer_interp"
# repo = "ARENA_materials"
# branch = "main"
#
# # Install dependencies
# try:
#     import jaxtyping
# except:
#     %pip install einops jaxtyping plotly
#
# # Get root directory, handling 3 different cases: (1) Colab, (2) notebook not in ARENA repo, (3) notebook in ARENA repo
# root = (
#     "/content"
#     if IN_COLAB
#     else "/root"
#     if repo not in os.getcwd()
#     else str(next(p for p in Path.cwd().parents if p.name == repo))
# )
#
# if Path(root).exists() and not Path(f"{root}/{chapter}").exists():
#     if not IN_COLAB:
#         !sudo apt-get install unzip
#         %pip install jupyter ipython --upgrade
#
#     if not os.path.exists(f"{root}/{chapter}"):
#         !wget -P {root} https://github.com/ARENA-education/ARENA_materials/archive/refs/heads/{branch}.zip
#         !unzip {root}/{branch}.zip '{repo}-{branch}/{chapter}/exercises/*' -d {root}
#         !mv {root}/{repo}-{branch}/{chapter} {root}/{chapter}
#         !rm {root}/{branch}.zip
#         !rmdir {root}/{repo}-{branch}
#
#
# if f"{root}/{chapter}/exercises" not in sys.path:
#     sys.path.append(f"{root}/{chapter}/exercises")
#
# os.chdir(f"{root}/{chapter}/exercises")

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import einops
import numpy as np
import plotly.express as px
import torch as t
from IPython.display import HTML, display
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm.auto import tqdm

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
section = "part55_param_decomp"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
# FILTERS: ~colab
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))
# END FILTERS

import part55_param_decomp.tests as tests
import part55_param_decomp.utils as utils
from plotly_utils import imshow, line

MAIN = __name__ == "__main__"

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []


r'''
# 1️⃣ Basics of Parameter Decomposition
'''


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''

We want to understand the "mechanisms" of a model, but how do we define mechanisms? Parameter decomposition is relatively minimal in what it presupposes: A typical network does not use all of it's machinery on every input. For an input like "2+2=", we do not need to recall the location of the eiffel tower, and this machinery should be idle for this input.

But how do we know what counts as idle machinery? Parameter decomposition offers one frame: We should be able to find pieces of parameters, such that some are causally important on an input, and others are not causally important on the input. 

We say a piece is **causally important** on an input if ablating it (fully or partially) changes the output of the model. The piece can only be said to be idle if keeping it on, or turning it off, or scaling to some intermediate value, has no effect on the output.

Parameter decomposition tries to split the model parameters into pieces of the parameters, such as to balance two opposing goals: a few pieces as possible are causally important, and as many unimportant pieces as of the model are ablatable, while keeping the input in tact.

'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Background: Singular Value Decomposition (SVD)

One of the oldest ways to decompose parameters in a model, is to do singular value decomposition.

Given the ($n \times n$) matrix of weights $W$ in a model layer, one can summarise the matrix as the sum of $n$ independent compontents, written as:

\[
$W$ = U S V^T = \sum_i \sigma_i ( u_i \cdot v_i^T) 
]\

</details>
<figure class="diagram">
<img src="https://cute.sus.cat/dev/img/svd-refresher.svg" alt="SVD as a sum of rank-1 terms, and what one term does to an input" width="860">
<figcaption>SVD two ways: the factorisation $W = U \Sigma V^\top$ as a sum of rank-1 terms, and the pipeline one term applies to an input ("project, scale, write").</figcaption>
</figure>

'''