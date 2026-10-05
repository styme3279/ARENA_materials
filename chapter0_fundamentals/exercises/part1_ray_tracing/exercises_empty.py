

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

# %%

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
from tqdm import tqdm

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
    full_arr = t.zeros((num_pixels, 2, 3))
    full_arr[:, 1, 0] = 1
    y_arr = t.linspace(-y_limit, y_limit, num_pixels)
    # print(f"y_arr: {y_arr}")
    full_arr[:, 1, 1] = y_arr

    return full_arr


rays1d = make_rays_1d(9, 10.0)
fig = render_lines_with_plotly(rays1d)

# %%

fig: go.FigureWidget = setup_widget_fig_ray()
display(fig)


@interact(v=(0.0, 6.0, 0.01), seed=(0, 10, 1))
def update(v=0.0, seed=0):
    t.manual_seed(seed)
    L_1, L_2 = t.rand(2, 2)
    P = lambda v: L_1 + v * (L_2 - L_1)
    x, y = zip(P(0), P(6))
    with fig.batch_update():
        fig.update_traces({"x": x, "y": y}, 0)
        fig.update_traces({"x": [L_1[0], L_2[0]], "y": [L_1[1], L_2[1]]}, 1)
        fig.update_traces({"x": [P(v)[0]], "y": [P(v)[1]]}, 2)
# %%

def intersect_ray_1d(ray: Float[Tensor, "points dims"], segment: Float[Tensor, "points dims"]) -> bool:
    """
    ray: shape (n_points=2, n_dim=3)  # O, D points
    [[0, 0, 0], [1, -1.0, 0]]
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points
    [[0, 0, 0], [1, -1.0, 0]]
    Return True if the ray intersects the segment.
    """

    # D_ray = ray[1, 1]
    # D_segment = (segment[1, 1] - segment[0, 1]) / (segment[1, 0] - segment[0, 0])
    
    Dx = ray[1,0] - ray[0, 0]
    Dy = ray[1,1] - ray[0, 1]

    l1minl2_x = (segment[1, 0] - segment[0, 0])
    l1minl2_y = (segment[1, 1] - segment[0, 1])
    l1min0_x = segment[1, 0]
    l1min0_y = segment[1, 1]

    lhs = t.tensor([[Dx, l1minl2_x], [Dy, l1minl2_y]])
    rhs = t.tensor([l1min0_x, l1min0_y])

    # print(f"lhs shape: {lhs.shape}")

    # print(f"lhs shape: {lhs.shape}")
    # print(f"rhs shape: {rhs.shape}")

    try:
        x = t.linalg.solve(lhs, rhs)
    except:
        return False
    # print(f"x shape: {x.shape}")

    if x[0] >= 0 and 0 <= x[1] <= 1:
        return True
    else:
        return False

tests.test_intersect_ray_1d(intersect_ray_1d)
tests.test_intersect_ray_1d_special_case(intersect_ray_1d)



# %%

def intersect_rays_1d(
    rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]
) -> Bool[Tensor, " nrays"]:
    """
    For each ray, return True if it intersects any segment.
    [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    """
    Dx = rays[:, 1, 0] - rays[:, 0, 0]
    Dy = rays[:, 1, 1] - rays[:, 0, 1]
    l1minl2_x = segments[:, 1, 0] - segments[:, 0, 0]
    l1minl2_y = segments[:, 1, 1] - segments[:, 0, 1]
    l1min0_x = segments[:, 1, 0]
    l1min0_y = segments[:, 1, 1]


tests.test_intersect_rays_1d(intersect_rays_1d)
tests.test_intersect_rays_1d_special_case(intersect_rays_1d)