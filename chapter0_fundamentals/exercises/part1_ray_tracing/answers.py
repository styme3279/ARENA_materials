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
    print(D.shape)
    print(a.shape)

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

# def intersect_rays_1d(
#     rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]
# ) -> Bool[Tensor, " nrays"]:
#     """
#     For each ray, return True if it intersects any segment.
#     """

#     O = rays[:, 0,:2]
#     D = rays[:, 1,:2]
#     L1 = segments[:, 0, :2]
#     L2 = segments[:, 1, :2]

#     # print(L1 - L2)
#     a = t.stack((D, L1-L2), dim=2)
#     # print(a)

#     b = t.unsqueeze(L1-O, dim=1).T
#     # print(b)

#     determinants = t.linalg.det(a)
#     print(a.shape)
#     print(determinants)
#     is_singular = determinants.abs() < 1e-6
#     print(is_singular)
#     print(a.shape)
#     print(a.size(dim=-1))
#     a[is_singular] = t.eye(a.size(dim=-1))

#     # try:
#     #     intersection = t.linalg.solve(a, b)
#     #     # print(intersection)
#     # except RuntimeError:
#     #     return False
#     # if intersection[0] >= 0.0 and intersection[1] >= 0.0 and intersection[1] <= 1.0:
#     #     return True
#     # else:
#     #     return False


# # intersect_rays_1d(t.rand(5, 2, 3), t.rand(5, 2, 3))



# tests.test_intersect_rays_1d(intersect_rays_1d)
# tests.test_intersect_rays_1d_special_case(intersect_rays_1d)

# %%

def intersect_rays_1d(
    rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]
) -> Bool[Tensor, " nrays"]:
    """
    For each ray, return True if it intersects any segment.
    """

    # Remove z dimension
    rays = rays[:,:,:-1]
    segments = segments[:,:,:-1]

    # Create every combination
    n_rays = rays.shape[0]
    n_segments = segments.shape[0]
    rays     = einops.repeat(rays,     "nrays h w -> nrays nsegs h w", nsegs=n_segments)
    segments = einops.repeat(segments, "nsegs h w -> nrays nsegs h w", nrays=n_rays)

    O = rays[:, :, 0]
    D = rays[:, :, 1]

    L_1 = segments[:, :, 0]
    L_2 = segments[:, :, 1]

    M = t.stack((D, L_1 - L_2), dim=-1)
    # v = t.unsqueeze(L_1-O, dim=1).T
    v = L_1-O
    # print(D.shape)
    # print(M.shape)
    # print(v.shape)

    M = einops.rearrange(M, "a b h w -> (a b) h w")
    # print(M.shape)
    determinants = t.linalg.det(M)
    is_singular = determinants.abs() < 1e-6
    M[is_singular] = t.eye(M.size(dim=-1))

    v = L_1 - O
    v = einops.rearrange(v, "a b c -> (a b) c")
    # v = t.unsqueeze(L_1-O, dim=1).T


    # print(M.shape, v.shape)

    intersections = t.linalg.solve(M, v)
    intersections = einops.rearrange(intersections, '(rays segments) x -> rays segments x',rays=n_rays)
    valid = (intersections[:,:,0] >= 0.0) & (intersections[:,:,1] >= 0) & (intersections[:,:,1] <= 1)

    return valid.any(dim=1)




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

    v0 = t.tensor([0.0, 0.0, 0.0])
    
    a = t.ones(num_pixels_y * num_pixels_z)
    
    b = t.linspace(-y_limit, y_limit, steps=num_pixels_y)
    c = t.linspace(-z_limit, z_limit, steps=num_pixels_z)
    
    b = einops.repeat(b, "ny -> (ny nz)", nz=num_pixels_z)
    c = einops.repeat(c, "nz -> (ny nz)", ny=num_pixels_y) 
    
    d = t.stack((a, b, c), dim=1)

    z = t.zeros(d.shape)

    final = t.stack((z, d), dim=1)

    # print(final)

    return final




# tests.test_make_rays_2d(make_rays_2d)

rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
render_lines_with_plotly(rays_2d)









    # repeat(ims[0], "h w c -> h new_axis w c", new_axis=5).shape

    # rays = t.concat([rays] * segments.shape[0])
    # segments = t.concat([segments] * n_rays)



    # print(rays.shape)
    # print(segments.shape)
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
    # print("START")
    # raise NotImplementedError()
    # print(ray)


    # L1 = segment[0,:2]
    # L2 = segment[1,:2]

    # print(L1 - L2)
    a = t.stack((-D, B-A, C-A), dim=1)
    # print(D.shape)
    # print(a.shape)

    # b = t.unsqueeze(O-A, dim=0).T
    b = O - A
    # print(b)

    try:
        intersection = t.linalg.solve(a, b)
        # print(intersection)
    except RuntimeError:
        return False
    #if intersection[0] >= 0.0 and intersection[1] >= 0.0 and intersection[1] <= 1.0:
    if intersection[0] >= 0.0 and\
       intersection[1] >= 0.0 and\
       intersection[2] >= 0.0 and\
       intersection[1] + intersection[2] <= 1.0:
        return True
    else:
        return False


tests.test_triangle_ray_intersects(triangle_ray_intersects)

# %%


# def raytrace_triangle(
#     rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
#     triangle: Float[Tensor, "trianglePoints=3 dims=3"],
# ) -> Bool[Tensor, " nrays"]:
#     """
#     For each ray, return True if the triangle intersects that ray.
#     """

#     # Remove z dimension
#     rays = rays[:,:,:-1]
#     segments = segments[:,:,:-1]

#     # Create every combination
#     n_rays = rays.shape[0]
#     rays   = einops.repeat(rays, "nrays h w -> nrays nsegs h w", nsegs=n_segments)
#     # segments = einops.repeat(segments, "nsegs h w -> nrays nsegs h w", nrays=n_rays)

#     O = rays[:, :, 0]
#     D = rays[:, :, 1]

#     L_1 = segments[:, :, 0]
#     L_2 = segments[:, :, 1]

#     # a = t.stack((-D, B-A, C-A), dim=1)
#     M = t.stack((-D, B-A, C-A), dim=-1)
#     # v = t.unsqueeze(L_1-O, dim=1).T
#     v = L_1-O
#     # print(D.shape)
#     # print(M.shape)
#     # print(v.shape)

#     M = einops.rearrange(M, "a b h w -> (a b) h w")
#     # print(M.shape)
#     determinants = t.linalg.det(M)
#     is_singular = determinants.abs() < 1e-6
#     M[is_singular] = t.eye(M.size(dim=-1))

#     v = L_1 - O
#     v = einops.rearrange(v, "a b c -> (a b) c")
#     # v = t.unsqueeze(L_1-O, dim=1).T


#     # print(M.shape, v.shape)

#     intersections = t.linalg.solve(M, v)
#     intersections = einops.rearrange(intersections, '(rays segments) x -> rays segments x',rays=n_rays)
#     valid = (intersections[:,:,0] >= 0.0) & (intersections[:,:,1] >= 0) & (intersections[:,:,1] <= 1)

#     return valid.any(dim=1)


def raytrace_triangle(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"],
) -> Bool[Tensor, " nrays"]:
    """
    For each ray, return True if the triangle intersects that ray.
    """
    # print(rays.shape)
    A, B, C = triangle
    A = einops.repeat(A, 'dims -> nrays dims', nrays=rays.size(0))
    B = einops.repeat(B, 'dims -> nrays dims', nrays=rays.size(0))
    C = einops.repeat(C, 'dims -> nrays dims', nrays=rays.size(0))

    O = rays[:,0,:]
    D = rays[:,1,:]

    M = t.stack((-D, B - A, C - A), dim=-1)
    v = O - A

    try:
        sols = t.linalg.solve(M, v)
    except RuntimeError:
        return False

    s = sols[:,0]
    u = sols[:,1]
    v = sols[:,2]

    return (s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1)


A = t.tensor([1, 0.0, -0.5])
B = t.tensor([1, -0.5, 0.0])
C = t.tensor([1, 0.5, 0.5])
num_pixels_y = num_pixels_z = 50
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


def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"],
) -> Float[Tensor, " nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    """

    # Create every combination
    n_rays = rays.shape[0]
    n_tris = triangles.shape[0]
    rays      = einops.repeat(rays,      "rays raypoints dims -> rays tris raypoints dims", tris=n_tris)
    triangles = einops.repeat(triangles, "tris tripoints dims -> rays tris tripoints dims", rays=n_rays)

    # O = rays[:, :, 0]
    # D = rays[:, :, 1]
    O, D = rays.unbind(dim=2)
    A, B, C = triangles.unbind(dim=2)

    print(A.shape)
    print(O.shape)
    M = t.stack((-D, B-A, C-A), dim=-1)
    # v = t.unsqueeze(L_1-O, dim=1).T
    # print(D.shape)
    # print(M.shape)
    # print(v.shape)

    M = einops.rearrange(M, "a b h w -> (a b) h w")
    print(M.shape)
    determinants = t.linalg.det(M)
    is_singular = determinants.abs() < 1e-6
    M[is_singular] = t.eye(M.size(dim=-1))

    # v = L_1 - O
    v = O-A
    v = einops.rearrange(v, "a b c -> (a b) c")
    # # v = t.unsqueeze(L_1-O, dim=1).T


    # # print(M.shape, v.shape)

    intersections = t.linalg.solve(M, v)
    intersections = einops.rearrange(intersections, '(rays segments) x -> rays segments x',rays=n_rays)
    # valid = (intersections[:,:,0] >= 0.0) & (intersections[:,:,1] >= 0) & (intersections[:,:,1] <= 1)
    valid = (intersections[:, :, 0] >= 0.0) &\
            (intersections[:, :, 1] >= 0.0) &\
            (intersections[:, :, 2] >= 0.0) &\
            (intersections[:, :, 1] + intersections[:, :, 2] <= 1.0)

    return valid.any(dim=1)


triangles = t.load(section_dir / "pikachu.pt", weights_only=True)

num_pixels_y = 120
num_pixels_z = 120
y_limit = z_limit = 1

rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
rays[:, 0] = t.tensor([-2, 0.0, 0.0])
dists = raytrace_mesh(rays, triangles)
intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
dists_square = dists.view(num_pixels_y, num_pixels_z)
img = t.stack([intersects, dists_square], dim=0)

fig = px.imshow(img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000)
fig.update_layout(coloraxis_showscale=False)
for i, text in enumerate(["Intersects", "Distance"]):
    fig.layout.annotations[i]["text"] = text
fig.show()



# %%
