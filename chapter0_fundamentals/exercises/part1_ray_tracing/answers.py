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

    # [nrays, nsegments, 2]
    B = L_1 - O

    # [nrays, nsegments, uv]
    sol = t.linalg.solve(A, B)

    # [nrays, nsegments]
    u = sol[..., 0]
    v = sol[..., 1]

    return ((u >= 0) & (v >= 0) & (v <= 1) & ~is_singular).any(dim=-1)

tests.test_intersect_rays_1d(intersect_rays_1d)
tests.test_intersect_rays_1d_special_case(intersect_rays_1d)

# %%
def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> Float[Tensor, "nrays 2 3"]:
    """
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    """
    nrays = num_pixels_y * num_pixels_z
    rays = t.zeros(nrays, 2, 3)
    ys = t.linspace(-y_limit, y_limit, num_pixels_y)
    zs = t.linspace(-z_limit, z_limit, num_pixels_z)
    ygrid = einops.repeat(ys, "y -> (y z)", z=num_pixels_z)
    zgrid = einops.repeat(zs, "z -> (y z)", y=num_pixels_y)

    rays[:, 1, 0] = 1
    rays[:, 1, 1] = ygrid
    rays[:, 1, 2] = zgrid
    return rays

rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
render_lines_with_plotly(rays_2d)

# %%
Point = Float[Tensor, "points=3"]

def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    """
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    """
    vec = O - A
    mat = t.stack((-D, B-A, C-A), dim=-1)
    try:
        sol = t.linalg.solve(mat, vec)
    except RuntimeError:
        return False
    s, u, v = sol
    return s >= 0 and u >= 0 and v >= 0 and u+v<=1

tests.test_triangle_ray_intersects(triangle_ray_intersects)

# %%
def raytrace_triangle(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"],
) -> Bool[Tensor, " nrays"]:
    """
    For each ray, return True if the triangle intersects that ray.
    """
    # [nrays, dims]
    O = rays[:, 0]
    D = rays[:, 1]
    # [dims]
    A, B, C = triangle
    # [nrays, dims]
    vec = O - A
    # [nrays, dims (3), 3]
    mat = t.stack((-D, B-A, C-A), dim=-1)
    # [nrays]
    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-6
    mat[is_singular] = t.eye(3)
    sol = t.linalg.solve(mat, vec)
    s, u, v = sol
    return ((u >= 0) & (v >= 0) & (v <= 1) & ~is_singular).any(dim=-1)

A = t.tensor([1, 0.0, -0.5])
B = t.tensor([1, -0.5, 0.0])
C = t.tensor([1, 0.5, 0.5])
num_pixels_y = num_pixels_z = 15
y_limit = z_limit = 0.5

# Plot triangle & rays
test_triangle = t.stack([A, B, C], dim=0)
rays2d = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
triangle_lines = t.stack([A, B, C, A, B, C], dim=0).reshape(-1, 2, 3)
render_lines_with_plotly(rays2d, triangle_lines)

# Calculate and display intersections
intersects = raytrace_triangle(rays2d, test_triangle)
img = intersects.reshape(num_pixels_y, num_pixels_z).int()
imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")

# %%
