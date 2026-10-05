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
    # return t.arange(start=-y_limit, end=y_limit, step=y_limit * 2 / num_pixels)
    out = t.zeros((num_pixels, 2, 3), dtype=t.float32)
    out[:, 1, 0] = 1
    ar = t.linspace(-y_limit, y_limit, num_pixels)
    out[:, 1, 1] = ar
    # print(out)
    # raise NotImplementedError()
    return out


rays1d = make_rays_1d(9, 10.0)
# fig = render_lines_with_plotly(rays1d)

def intersect_ray_1d(ray: Float[Tensor, "points dims"], segment: Float[Tensor, "points dims"]) -> bool:
    """
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    """
    O, D = ray[:, :2]
    L1, L2 = segment[:, :2]
    A = t.stack([D, L1 - L2], dim=1)
    B = L1 - O
    try:
        X = t.linalg.solve(A, B)
    except t.linalg.LinAlgError:
        return False

    u, v = X
    return u >= 0 and 0 <= v <= 1
   

# tests.test_intersect_ray_1d(intersect_ray_1d)
# tests.test_intersect_ray_1d_special_case(intersect_ray_1d)

def intersect_rays_1d(
    rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]
) -> Bool[Tensor, " nrays"]:
    """
    For each ray, return True if it intersects any segment.
    """
    # breakpoint()
    Os, Ds = einops.rearrange(rays[..., :2], "n p d -> p n d")
    L1s, L2s = einops.rearrange(segments[..., :2], "n p d -> p n d")
    D_grid = einops.repeat(Ds, "nr d -> nr ns d", ns=segments.shape[0])
    O_grid = einops.repeat(Os, "nr d -> nr ns d", ns=segments.shape[0])
    L1_grid = einops.repeat(L1s, "ns d -> nr ns d", nr=rays.shape[0])
    L2_grid = einops.repeat(L2s, "ns d -> nr ns d", nr=rays.shape[0])
    As = t.stack([D_grid, L1_grid - L2_grid], dim=-1)
    print(As)
    print(As.shape)

    Bs = L1_grid - O_grid
    dets = t.linalg.det(As)
    print(dets)
    print(dets.shape)

    is_singular = dets.abs() < 1e-8
    As[is_singular] = t.eye(2)

    X = t.linalg.solve(As, Bs)
    print(X)
    print(X.shape)

    # X[~is_singular].any()

    u, v = X.unbind(dim=-1)
    print(u.shape, v.shape)
    # breakpoint()
    hits = (u >= 0) & (0 <= v) & (v <= 1) & ~is_singular
    return hits.any(dim=-1)

    # return


    # L1s, L2s, = einops.rearrange(segments[..., :2], "n p d -> p n 1 d")
    # As = einops.rearrange([Ds, L1s - L2s], "n d p d -> n (d, p d)")
    # As = t.stack([Ds, L1s - L2s], dim=-1)


    # L1, L2 = segment[:, :2]
    #     A = t.stack([D, L1 - L2], dim=1)
    #     B = L1 - O
    #     try:
    #         X = t.linalg.solve(A, B)
    #     except t.linalg.LinAlgError:
    #         return False
    
    #     u, v = X
    #     return u >= 0 and 0 <= v <= 1

    # print(Os, Ds)
    # raise NotImplementedError()


tests.test_intersect_rays_1d(intersect_rays_1d)
tests.test_intersect_rays_1d_special_case(intersect_rays_1d)

def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> Float[Tensor, "nrays 2 3"]:
    """
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    """
    # breakpoint()
    out = t.zeros((num_pixels_y * num_pixels_z, 2, 3), dtype=t.float32)
    out[:, 1, 0] = 1
    ar_y = t.linspace(-y_limit, y_limit, num_pixels_y)
    ar_z = t.linspace(-z_limit, z_limit, num_pixels_z)
    out[:, 1, 1] = einops.repeat(ar_y, "y -> (y z)", z=num_pixels_z)
    out[:, 1, 2] = einops.repeat(ar_z, "z -> (y z)", y=num_pixels_y)
    # print(out)
    # raise NotImplementedError()
    return out
    # raise NotImplementedError()


rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
# render_lines_with_plotly(rays_2d)

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
    mat = t.stack([-D, B-A, C-A], dim=-1)
    vec = O-A
    try:
        X = t.linalg.solve(mat, vec)
    except t.linalg.LinAlgError:
        return False

    s, u, v = X
    # breakpoint()
    return s >= 0 and u >= 0 and v >= 0 and u + v <= 1


tests.test_triangle_ray_intersects(triangle_ray_intersects)

def raytrace_triangle(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"],
) -> Bool[Tensor, " nrays"]:
    """
    For each ray, return True if the triangle intersects that ray.
    """
    nrays = rays.shape[0]
    Os, Ds = einops.rearrange(rays, "n p d -> p n d")
    A, B, C = triangle.unbind(dim=0)
    As = einops.repeat(A, "d -> nr d", nr=nrays)
    Bs = einops.repeat(B, "d -> nr d", nr=nrays)
    Cs = einops.repeat(C, "d -> nr d", nr=nrays)
    mat = t.stack([-Ds, Bs-As, Cs-As], dim=-1)
    vec = Os-As
    
    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    X = t.linalg.solve(mat, vec)
    s, u, v = X.unbind(dim=-1)

    hits = (s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular
    return hits
    # raise NotImplementedError()


# A = t.tensor([1, 0.0, -0.5])
# B = t.tensor([1, -0.5, 0.0])
# C = t.tensor([1, 0.5, 0.5])
# num_pixels_y = num_pixels_z = 150
# y_limit = z_limit = 0.5

# # Plot triangle & rays
# test_triangle = t.stack([A, B, C], dim=0)
# rays2d = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
# triangle_lines = t.stack([A, B, C, A, B, C], dim=0).reshape(-1, 2, 3)
# render_lines_with_plotly(rays2d, triangle_lines)

# # Calculate and display intersections
# intersects = raytrace_triangle(rays2d, test_triangle)
# img = intersects.reshape(num_pixels_y, num_pixels_z).int()
# imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")

triangles = t.load(section_dir / "pikachu.pt", weights_only=True)
print(triangles.shape)

def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"],
) -> Float[Tensor, " nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    """

    # raise NotImplementedError()
    nrays = rays.shape[0]
    ntris = triangles.shape[0]
    Os, Ds = einops.rearrange(rays, "n p d -> p n d")
    A, B, C = einops.rearrange(triangles, "n p d -> p n d")
    As = einops.repeat(A, "nt d -> nr nt d", nr=nrays)
    Bs = einops.repeat(B, "nt d -> nr nt d", nr=nrays)
    Cs = einops.repeat(C, "nt d -> nr nt d", nr=nrays)
    Os, Ds = einops.repeat(Os, "nr d -> nr nt d", nt=ntris), einops.repeat(Ds, "nr d -> nr nt d", nt=ntris) 
    mat = t.stack([-Ds, Bs-As, Cs-As], dim=-1)
    vec = Os-As
    
    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    X = t.linalg.solve(mat, vec)
    s, u, v = X.unbind(dim=-1)

    hits = (s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular
    # breakpoint()

    dists = t.where(hits, s, t.inf)
    dists2 = dists.min(dim=-1).values
    # breakpoint()
    return dists2

    # return t.Tensor([100.0 if e is True else float('inf') for e in hits.any(dim=1)])


num_pixels_y = 120
num_pixels_z = 120
y_limit = z_limit = 1

rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
rays[:, 0] = t.tensor([-2.5, 0.5, 0.0])
dists = raytrace_mesh(rays, triangles)
intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
dists_square = dists.view(num_pixels_y, num_pixels_z)
img = t.stack([intersects, dists_square], dim=0)

fig = px.imshow(img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000)
fig.update_layout(coloraxis_showscale=False)
for i, text in enumerate(["Intersects", "Distance"]):
    fig.layout.annotations[i]["text"] = text
fig.show()

def rotation_matrix(theta: Float[Tensor, ""]) -> Float[Tensor, "rows cols"]:
    """
    Creates a rotation matrix representing a counterclockwise rotation of `theta` around the y-axis.
    """
    raise NotImplementedError()


tests.test_rotation_matrix(rotation_matrix)