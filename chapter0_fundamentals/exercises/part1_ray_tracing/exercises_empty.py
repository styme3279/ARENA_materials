

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

    # print(f"t.stack([Dx,Dy]: {t.stack([Dx,Dy],dim=1)}")

    lhs_d = einops.repeat(t.stack([Dx,Dy],dim=1),"nrays w ->  (nrays nsegments) w", nsegments=segments.shape[0])
    lhs_l = einops.repeat(t.stack([l1minl2_x,l1minl2_y],dim=1),"nsegments w-> (nrays nsegments) w", 
    nrays=rays.shape[0])


    lhs = t.concat([lhs_d, lhs_l], dim=1)
    # print(f"lhs: {lhs}")
    lhs = einops.rearrange(lhs, 'i (k l) -> i l k', k=2, l=2)
    # print(f"lhs: {lhs}")

    # dets = lhs.det()
    # lhs = [lhs[i] if dets[i] > 1e-8 else t.eye(2) for i in range(len(dets))]
    # print(f"lhs: {lhs}")
    dets = t.linalg.det(lhs)
    # print(f"dets: {dets}")
    is_singular = dets.abs() < 1e-8
    # print(f"is_singular: {is_singular}")
    # print(f"lhs.shape: {lhs.shape}")
    # print(f"is_singular.shape: {is_singular.shape}")

    lhs[is_singular] = t.eye(2)
    # print(f"lhs: {lhs}")

    new_tensor = t.stack([l1min0_x, l1min0_y], dim=1)
    rhs = einops.repeat(new_tensor, "nsegments w-> (nrays nsegments) w", nrays=rays.shape[0])

    x = t.linalg.solve(lhs, rhs)
    # print(f"x[:,1]: {x[:,1]}")
    intersect = x[:,0] >= 0  #t.any(x[:,0] >= 0)
    # print(f"intersect: {intersect}")
    
    valid_l =  x[:,1] >= 0
    valid_r =  x[:,1] <= 1
    valid = valid_l * valid_r

    both = intersect * valid
    both = einops.rearrange(both, "(n w) -> n w", w = segments.shape[0])

    any_overlap = t.sum(both, dim=1).bool()

    return any_overlap

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
    full_arr = t.zeros((num_pixels_y*num_pixels_z, 2, 3))
    full_arr[:, 1, 0] = 1
    y_arr = t.linspace(-y_limit, y_limit, num_pixels_y)
    z_arr = t.linspace(-z_limit, z_limit, num_pixels_z)

    yz_grid = t.cartesian_prod(y_arr,z_arr)
    full_arr[:, 1, 1:] = yz_grid

    return full_arr




rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
render_lines_with_plotly(rays_2d)


# %%

one_triangle = t.tensor([[0, 0, 0], [4, 0.5, 0], [2, 3, 0]])
A, B, C = one_triangle
x, y, z = one_triangle.T

fig: go.FigureWidget = setup_widget_fig_triangle(x, y, z)
display(fig)


@interact(u=(-0.5, 1.5, 0.01), v=(-0.5, 1.5, 0.01))
def update(u=0.0, v=0.0):
    P = A + u * (B - A) + v * (C - A)
    fig.update_traces({"x": [P[0]], "y": [P[1]]}, 2)
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

    Dx, Dy, Dz = D[0], D[1], D[2]
    BAx, BAy, BAz = B[0] - A[0], B[1] - A[1], B[2] - A[2]
    CAx, CAy, CAz = C[0] - A[0], C[1] - A[1], C[2] - A[2]
    OAx, OAy, OAz = O[0] - A[0], O[1] - A[1], O[2] - A[2]

    lhs = t.tensor([[-Dx,BAx,CAx]])
    rh
tests.test_triangle_ray_intersects(triangle_ray_intersects)

def intersect_ray_1d(ray: Float[Tensor, "points dims"], segment: Float[Tensor, "points dims"]) -> bool:
    """
    ray: shape (n_points=2, n_dim=3)  # O, D points
    [[0, 0, 0], [1, -1.0, 0]]
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points
    [[0, 0, 0], [1, -1.0, 0]]
    Return True if the ray intersects the segment.
    """
    Dx = ray[1,0] - ray[0, 0]
    Dy = ray[1,1] - ray[0, 1]
    l1minl2_x = (segment[1, 0] - segment[0, 0])
    l1minl2_y = (segment[1, 1] - segment[0, 1])
    l1min0_x = segment[1, 0]
    l1min0_y = segment[1, 1]
    lhs = t.tensor([[Dx, l1minl2_x], [Dy, l1minl2_y]])
    rhs = t.tensor([l1min0_x, l1min0_y])
    try:
        x = t.linalg.solve(lhs, rhs)
    except:
        return False
    if x[0] >= 0 and 0 <= x[1] <= 1:
        return True
    else:
        return False