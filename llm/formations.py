"""
Geometric formation library.

These functions turn a formation name + drone count into 3D target positions.
This is the deterministic geometry the LLM commander will *select and
parameterize* (Layer 1). Keeping it as a clean library means:
  - the demo runs without any API key,
  - the LLM's job is reduced to "pick a formation + params", which is
    reliable and easy to validate.

Each function returns an (n, 3) array of world targets.
"""
import numpy as np

DEFAULT_Z = 1.5


def circle(n, radius=2.0, center=(0, 0), z=DEFAULT_Z):
    cx, cy = center
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + radius * np.cos(a), cy + radius * np.sin(a), np.full(n, z)], axis=1)


def star(n, outer=2.4, inner=1.0, center=(0, 0), z=DEFAULT_Z):
    """Alternating outer/inner radius -> classic star-polygon outline."""
    cx, cy = center
    a = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
    r = np.where(np.arange(n) % 2 == 0, outer, inner)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a), np.full(n, z)], axis=1)


def line(n, spacing=0.6, center=(0, 0), z=DEFAULT_Z, axis="x"):
    cx, cy = center
    offs = (np.arange(n) - (n - 1) / 2) * spacing
    if axis == "x":
        return np.stack([cx + offs, np.full(n, cy), np.full(n, z)], axis=1)
    return np.stack([np.full(n, cx), cy + offs, np.full(n, z)], axis=1)


def vee(n, spacing=0.6, angle_deg=45, center=(0, 0), z=DEFAULT_Z):
    cx, cy = center
    th = np.radians(angle_deg)
    pts = [(cx, cy, z)]
    for k in range(1, n):
        side = 1 if k % 2 == 1 else -1
        idx = (k + 1) // 2
        pts.append((cx + side * idx * spacing * np.sin(th),
                    cy - idx * spacing * np.cos(th), z))
    return np.array(pts[:n])


def grid(n, spacing=0.7, center=(0, 0), z=DEFAULT_Z):
    cols = int(np.ceil(np.sqrt(n)))
    pts = []
    for i in range(n):
        r, c = divmod(i, cols)
        pts.append((center[0] + (c - (cols - 1) / 2) * spacing,
                    center[1] + (r - (cols - 1) / 2) * spacing, z))
    return np.array(pts)


def square(n, side=3.0, center=(0, 0), z=DEFAULT_Z):
    """Drones spread evenly around the outline (perimeter) of a square."""
    cx, cy = center
    h = side / 2.0
    # 4 corners, walked clockwise; sample the perimeter at n equal arc-lengths.
    corners = np.array([(-h, -h), (-h, h), (h, h), (h, -h)])
    per = np.linspace(0, 4, n, endpoint=False)        # which edge + fraction
    pts = []
    for p in per:
        e = int(p) % 4
        f = p - int(p)
        a, b = corners[e], corners[(e + 1) % 4]
        x, y = a + f * (b - a)
        pts.append((cx + x, cy + y, z))
    return np.array(pts)


def heart(n, size=4.0, center=(0, 0), z=DEFAULT_Z):
    """Drones evenly spaced (by arc length) along a parametric heart curve."""
    cx, cy = center
    tt = np.linspace(0, 2 * np.pi, 400)
    x = 16 * np.sin(tt) ** 3
    y = (13 * np.cos(tt) - 5 * np.cos(2 * tt)
         - 2 * np.cos(3 * tt) - np.cos(4 * tt))
    # resample at n equal arc-length positions so drones don't bunch at the cusp
    seg = np.hypot(np.diff(x), np.diff(y))
    cum = np.concatenate([[0], np.cumsum(seg)])
    want = np.linspace(0, cum[-1], n, endpoint=False)
    xi, yi = np.interp(want, cum, x), np.interp(want, cum, y)
    s = size / 32.0                      # raw curve spans ~32 units wide
    return np.stack([cx + s * xi, cy + s * yi, np.full(n, z)], axis=1)


def fibonacci(n, scale=0.78, center=(0, 0), z=DEFAULT_Z):
    """Fibonacci / phyllotaxis spiral (sunflower seed-head): drones at golden-angle
    increments with radius ~ sqrt(index) -> a naturally even, swirling disc."""
    cx, cy = center
    golden = np.pi * (3 - np.sqrt(5))                # ~137.5 deg
    i = np.arange(n)
    r = scale * np.sqrt(i + 0.5)
    a = i * golden
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a), np.full(n, z)], axis=1)


REGISTRY = {"circle": circle, "star": star, "line": line, "vee": vee,
            "grid": grid, "square": square, "heart": heart, "fibonacci": fibonacci}

MIN_SEP = 1.5            # metres: no two drones may be closer than this


def _enforce_min_sep(pts, min_sep=MIN_SEP):
    """Uniformly scale the formation about its centroid until the closest pair of
    drones is at least ``min_sep`` apart (a hard no-fly bubble around each drone)."""
    if len(pts) < 2:
        return pts
    xy = pts[:, :2]
    diff = xy[:, None, :] - xy[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
    dmin = dist.min()
    if dmin >= min_sep or dmin == 0:
        return pts
    c = xy.mean(axis=0)
    scale = min_sep / dmin
    out = pts.copy()
    out[:, :2] = c + (xy - c) * scale
    return out


def make(name, n, min_sep=MIN_SEP, **kwargs):
    if name not in REGISTRY:
        raise ValueError(f"unknown formation '{name}'. options: {list(REGISTRY)}")
    pts = REGISTRY[name](n, **kwargs)
    return _enforce_min_sep(pts, min_sep)


def assign_targets(start_xy, target_xy):
    """Collision-free slot assignment by ANGULAR ORDER: sort the drones and the
    formation slots by angle about their centroids, then map the k-th drone to the
    k-th slot. Drones expand radially into the shape without crossing paths (no
    swaps), so they reach their places without colliding. Returns slot index per
    drone."""
    s, t = np.asarray(start_xy)[:, :2], np.asarray(target_xy)[:, :2]
    a_s = np.arctan2(*(s - s.mean(0))[:, ::-1].T)
    a_t = np.arctan2(*(t - t.mean(0))[:, ::-1].T)
    order_s, order_t = np.argsort(a_s), np.argsort(a_t)
    perm = np.empty(len(s), dtype=int)
    perm[order_s] = order_t
    return perm
