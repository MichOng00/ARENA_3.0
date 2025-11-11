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

#%%
# exercise 1: rays
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
    step = y_limit/num_pixels
    arr = []
    for i in range(num_pixels):
        arr.append([[0,0,0], [1, -1+step*i, 0]])
    return t.tensor(arr)


rays1d = make_rays_1d(9, 10.0)
fig = render_lines_with_plotly(rays1d)
# %%
def intersect_ray_1d(
    ray: Float[Tensor, "points dims"], segment: Float[Tensor, "points dims"]
) -> bool:
    """
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    """
    O, D = ray[:, :2]
    L1, L2 = segment[:, :2]

    matrix = t.stack([D, (L1-L2)], dim=-1)
    vector = t.tensor(L1-O)

    try:
        soln = t.linalg.solve(matrix, vector)
    except:
        return False
    return (soln[0] >= 0) and (0 <= soln[1] <= 1)

tests.test_intersect_ray_1d(intersect_ray_1d)
tests.test_intersect_ray_1d_special_case(intersect_ray_1d)

# %%
def intersect_rays_1d(
    rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]
) -> Bool[Tensor, "nrays"]:
    """
    For each ray, return True if it intersects any segment.
    """
    num_rays = rays.size(0)
    num_segments = segments.size(0)
    answer = t.zeros(num_rays)

    ray_coords = rays[..., :2]
    segment_coords = segments[..., :2]

    ray_coords = einops.repeat(ray_coords, "a b c-> a d b c", d=num_segments)
    segment_coords = einops.repeat(segment_coords, "a b c-> d a b c", d=num_rays)

    O = ray_coords[:, :, 0]
    D = ray_coords[:, :, 1]
    L1 = segment_coords[:, :, 0]
    L2 = segment_coords[:,:, 1]

    matrix = t.stack([D, (L1-L2)], dim=-1)
    vector = L1-O

    dets = t.linalg.det(matrix)
    is_singular = dets.abs() < 1e-8
    matrix[is_singular] = t.eye(2)

    soln = t.linalg.solve(matrix, vector)
    
    return ((soln[..., 0] >= 0) & (0 <= soln[..., 1]) & (soln[..., 1] <= 1) & (~is_singular)).any(dim=-1)


tests.test_intersect_rays_1d(intersect_rays_1d)
tests.test_intersect_rays_1d_special_case(intersect_rays_1d)
# %%
def make_rays_2d(
    num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float
) -> Float[Tensor, "nrays 2 3"]:
    """
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    """
    # rays = t.zeros((num_pixels, 2, 3), dtype=t.float32)
    # t.linspace(-y_limit, y_limit, num_pixels, out=rays[:, 1, 1])
    # rays[:, 1, 0] = 1
    # return rays
    num_pixels = num_pixels_y*num_pixels_z
    soln = t.zeros(num_pixels, 2, 3, dtype=t.float32)
    ys = t.linspace(-y_limit, y_limit, num_pixels_y)
    zs = t.linspace(-z_limit, z_limit, num_pixels_z)
    # soln[:, 0] is always origin
    soln[:, 1, 0] = 1 # x-coord of 2nd pt
    soln[:, 1, 1] = einops.repeat(ys, "y -> (y z)", z=num_pixels_z)
    soln[:, 1, 2] = einops.repeat(zs, "z -> (y z)", y=num_pixels_y)
    return soln


rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
render_lines_with_plotly(rays_2d)

#%%
soln = t.zeros(10, 2, 3, dtype=t.float32)
# print(soln)
soln[:, 1, 0] = 1
# print(soln)
ys = t.linspace(-2, 2, 5)
zs = t.linspace(-4, 4, 2)
soln[:, 1, 1] = einops.repeat(ys, "y -> (y z)", z=2)
print(ys)

#%%
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
    diffAB = B - A
    diffCA = C - A
    diffOA = O - A

    mat = t.stack((-D, diffAB, diffCA), dim=-1)
    try:
        s, u, v = t.linalg.solve(mat, diffOA)
    except:
        return False

    return ((s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1)).item()

# one_triangle = t.tensor([[0, 0, 0], [4, 0.5, 0], [2, 3, 0]])
# A, B, C = one_triangle
# ray = t.tensor([[0,0,0], [1, 2, 3]])
# O, D = ray
# triangle_ray_intersects(A, B, C, O, D)
tests.test_triangle_ray_intersects(triangle_ray_intersects)

#%%
def raytrace_triangle(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"],
) -> Bool[Tensor, "nrays"]:
    """
    For each ray, return True if the triangle intersects that ray.
    """
    num_rays = rays.size(0)
    A, B, C = einops.repeat(triangle, "pts dims -> pts NR dims", NR=num_rays)
    O, D = rays.unbind(dim=1)

    matrix = t.stack([-D, B-A, C-A], dim=-1)
    dets = t.linalg.det(matrix)
    is_singular = dets.abs()< 1e-8
    matrix[is_singular] = t.eye(3)

    vector = O-A
    soln = t.linalg.solve(matrix, vector)
    s, u, v = soln.unbind(dim=-1)
    
    return (~is_singular) & (u >= 0) & (v >= 0 )& (u+v<=1) & (s>=0)


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
triangles = t.load(section_dir / "pikachu.pt", weights_only=True)

def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"],
) -> Float[Tensor, "nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    """
    num_rays = rays.size(0)
    num_triangles = triangles.size(0)
    A, B, C = einops.repeat(triangles, "NT pts dims -> pts NR NT dims", NR=num_rays)
    O, D = einops.repeat(rays, "NR pts dims -> pts NR NT dims", NT=num_triangles)

    matrix = t.stack([-D, B-A, C-A], dim=-1)
    dets = t.linalg.det(matrix)
    is_singular = dets.abs()< 1e-8
    matrix[is_singular] = t.eye(3)

    vector = O-A
    soln = t.linalg.solve(matrix, vector)
    s, u, v = soln.unbind(dim=-1)

    s *= D[..., 0]
    
    intersects = (~is_singular) & (u >= 0) & (v >= 0 )& (u+v<=1) & (s>=0)
    s[~intersects] = float("inf")

    return einops.reduce(s, "NR NT -> NR", "min")


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