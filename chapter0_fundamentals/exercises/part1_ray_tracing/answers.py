# %%

import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

chapter = "chapter0_fundamentals"
repo = "ARENA_3.0"
branch = "main"

# Install dependencies
try:
    import jaxtyping
except:
    %pip install jaxtyping einops

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
        !wget -P {root} https://github.com/callummcdougall/ARENA_3.0/archive/refs/heads/{branch}.zip
        !unzip {root}/{branch}.zip '{repo}-{branch}/{chapter}/exercises/*' -d {root}
        !mv {root}/{repo}-{branch}/{chapter} {root}/{chapter}
        !rm {root}/{branch}.zip
        !rmdir {root}/{repo}-{branch}


if f"{root}/{chapter}/exercises" not in sys.path:
    sys.path.append(f"{root}/{chapter}/exercises")

os.chdir(f"{root}/{chapter}/exercises")

#%%
import os
import sys
from functools import partial
from pathlib import Path
from typing import Callable

import einops
import plotly.express as px
import plotly.graph_objects as go
import torch as t
from IPython.display import display
from ipywidgets import interact
from jaxtyping import Bool, Float
from torch import Tensor
from tqdm.auto import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part1_ray_tracing"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part1_ray_tracing.tests as tests
from part1_ray_tracing.utils import (
    render_lines_with_plotly,
    setup_widget_fig_ray,
    setup_widget_fig_triangle,
)
from plotly_utils import imshow

MAIN = __name__ == "__main__"

# %%


def make_rays_1d(num_pixels: int, y_limit: float) -> Tensor:
    """
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is
        also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains
        (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    """
    # raise NotImplementedError()
    v0 = t.tensor([0.0, 0.0, 0.0])

    a = t.ones(num_pixels)
    b = t.linspace(-y_limit, y_limit, steps=num_pixels)
    c = t.zeros(num_pixels)
    
    d = t.stack((a, b, c), dim=1)
    z = t.zeros(d.shape)

    final = t.stack((z, d), dim=1)

    # print(final)

    return final



# print(make_rays_1d(9, 1.0))

# tests.test_make_rays_1d(make_rays_1d)

rays1d = make_rays_1d(9, 10.0)
fig = render_lines_with_plotly(rays1d)

# %%

def intersect_ray_1d(ray: Float[Tensor, "points dims"], segment: Float[Tensor, "points dims"]) -> bool:
    """
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    """
    # raise NotImplementedError()
    # print(ray)

    O = ray[0,:2]
    D = ray[1,:2]
    L1 = segment[0,:2]
    L2 = segment[1,:2]

    # print(L1 - L2)
    a = t.stack((D, L1-L2), dim=1)
    # print(a)

    b = t.unsqueeze(L1-O, dim=0).T
    # print(b)

    try:
        intersection = t.linalg.solve(a, b)
        # print(intersection)
    except RuntimeError:
        return False
    if intersection[0] >= 0.0 and intersection[1] >= 0.0 and intersection[1] <= 1.0:
        return True
    else:
        return False


    # return None



# intersect_ray_1d(t.rand(2, 3), t.rand(2, 3))

tests.test_intersect_ray_1d(intersect_ray_1d)
tests.test_intersect_ray_1d_special_case(intersect_ray_1d)


# %%

def intersect_rays_1d(
    rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]
) -> Bool[Tensor, " nrays"]:
    """
    For each ray, return True if it intersects any segment.
    """

    O = rays[:, 0,:2]
    D = rays[:, 1,:2]
    L1 = segments[:, 0, :2]
    L2 = segments[:, 1, :2]

    # print(L1 - L2)
    a = t.stack((D, L1-L2), dim=2)
    # print(a)

    b = t.unsqueeze(L1-O, dim=1).T
    # print(b)

    determinants = t.linalg.det(a)
    print(a.shape)
    print(determinants)
    is_singular = determinants.abs() < 1e-6
    print(is_singular)
    print(a.shape)
    print(a[0, 0, :].shape)
    a[is_singular] = t.eye(a[0, 0, :].shape)

    # try:
    #     intersection = t.linalg.solve(a, b)
    #     # print(intersection)
    # except RuntimeError:
    #     return False
    # if intersection[0] >= 0.0 and intersection[1] >= 0.0 and intersection[1] <= 1.0:
    #     return True
    # else:
    #     return False


intersect_rays_1d(t.rand(5, 2, 3), t.rand(5, 2, 3))



# tests.test_intersect_rays_1d(intersect_rays_1d)
# tests.test_intersect_rays_1d_special_case(intersect_rays_1d)

# %%
