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
    rays = t.zeros(num_pixels, 2, 3)
    t.linspace(-y_limit, y_limit, num_pixels, out=rays[:, 1, 1])
    rays[:,1,0] = 1
    return rays

rays1d = make_rays_1d(9, 10.0)
fig = render_lines_with_plotly(rays1d)

# %%
def intersect_ray_1d(ray: Float[Tensor, "points dims"], segment: Float[Tensor, "points dims"]) -> bool:
    """
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    """
    ray = ray[:, :2]
    segment = ray[:, :2]
    O, D = ray
    L_1, L_2 = segment
    A = t.stack(D, L_1-L_2, dim=-1)
    B = L_1 - O
    try:
        sol = t.linalg.solve(A, B)
    except RuntimeError:
        return False

    u = sol[0].item()
    v = sol[1].item()
    return (u >= 0.0) and (v >= 0.0) and (v <= 1.0)


# %%
def intersect_rays_1d(
    rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]
) -> Bool[Tensor, " nrays"]:
    """
    For each ray, return True if it intersects any segment.
    """
    NR = rays.size(0)
    NS = segments.size(0)

    # [nrays, 2, 2]
    rays = rays[:, :, :2]
    # [nsegments, 2, 2]
    segments = segments[:, :, :2]

    # [nrays, nsegments, 2, 2]
    rays = einops.repeat(rays, "nrays point coord -> nrays nsegments point coord", nsegments=NS)
    segments = einops.repeat(segments, "nsegments point coord -> nrays nsegments point coord", nrays=NR)

    # [nrays, nsegments, 2]
    O = rays[:, :, 0, :]
    # [nrays, nsegments, 2]
    D = rays[:, :, 1, :]

    # [nrays, nsegments, 2]
    L_1 = segments[:, :, 0, :]
    # [nrays, nsegments, 2]
    L_2 = segments[:, :, 1, :]

    # [nrays, nsegments, 2, 2]
    A = t.stack([D, L_1-L_2], dim=-1)
    # [nrays, nsegments]
    dets = t.linalg.det(A)
    is_singular = dets.abs() < 1e-6

    A[is_singular] = t.eye(2)

    B = L_1 - O
    
    sol = t.linalg.solve(A, B)

    print(sol.shape)

tests.test_intersect_rays_1d(intersect_rays_1d)
tests.test_intersect_rays_1d_special_case(intersect_rays_1d)

# %%
