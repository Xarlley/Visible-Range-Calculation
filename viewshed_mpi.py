#!/usr/bin/env python3
"""
Parallel viewshed calculator (angle-bucket skyline), using mpi4py.
Run: mpiexec -n 8 python viewshed_mpi.py
"""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import array_bounds, rowcol
from mpi4py import MPI

# ---------- MPI Init ----------
comm  = MPI.COMM_WORLD
rank  = comm.Get_rank()
size  = comm.Get_size()

EPS      = 1e-9
ANGLE_RES = 0.5            # degree
N_BUCKET  = int(360/ANGLE_RES)  # 720

# ---------- 0. Load DEM (rank 0) ----------
if rank == 0:
    data_dir = Path("./data")
    tifs = sorted(data_dir.glob("*.tif"))
    if not tifs:
        raise FileNotFoundError("./data 内没有 .tif")
    srcs = [rasterio.open(p) for p in tifs]
    mosaic_m, transform = merge(srcs, masked=True)
    dem = mosaic_m[0].filled(0).astype("float32")
    rows, cols = dem.shape
    for s in srcs:
        s.close()
    left, bottom, right, top = array_bounds(rows, cols, transform)
    # 与用户交互
    print(f"坐标范围：X∈[{left:.2f},{right:.2f}]  Y∈[{bottom:.2f},{top:.2f}]")
    dx, dy = abs(transform.a), abs(transform.e)
    print(f"像元粒度：ΔX≈{dx:.3f}  ΔY≈{dy:.3f}")
    obs_x = float(input("请输入观察者 X 坐标: "))
    obs_y = float(input("请输入观察者 Y 坐标: "))
    R     = float(input("请输入扫描半径 R: "))
else:
    dem = None
    transform = None
    rows = cols = None
    obs_x = obs_y = R = None
    left = bottom = right = top = None

# ---------- 1. Broadcast shared data ----------
transform = comm.bcast(transform, root=0)
rows = comm.bcast(rows, root=0)
cols = comm.bcast(cols, root=0)
obs_x = comm.bcast(obs_x, root=0)
obs_y = comm.bcast(obs_y, root=0)
R     = comm.bcast(R,     root=0)

if rank != 0:
    dem = np.empty((rows, cols), dtype="float32")
comm.Bcast(dem, root=0)

# ---------- 2. Pre-compute distance & angle for radius mask ----------
rr, cc = np.indices((rows, cols))
obs_row, obs_col = rowcol(transform, obs_x, obs_y)

dx_pix = (cc - obs_col) * transform.a
dy_pix = (rr - obs_row) * -transform.e
dist   = np.hypot(dx_pix, dy_pix)
within = dist <= R

# 角度桶 [0, 720)
angles_idx = ( (np.degrees(np.arctan2(dy_pix, dx_pix)) + 360.0) % 360.0 ) / ANGLE_RES
angles_idx = angles_idx.astype("int16")

cand_rows  = rr[within]
cand_cols  = cc[within]
cand_dist  = dist[within]
cand_ang   = angles_idx[within]

# 距离升序排序
order = np.argsort(cand_dist)
cand_rows = cand_rows[order]
cand_cols = cand_cols[order]
cand_dist = cand_dist[order]
cand_ang  = cand_ang[order]

# ---------- 3. Local skyline per process ----------
visible_local = np.zeros_like(dem, dtype="uint8")   # 0/1
max_slope     = np.full(N_BUCKET, -np.inf, dtype="float32")

obs_elev = dem[obs_row, obs_col]

# 仅处理自己负责的桶
mask_proc = (cand_ang % size) == rank
sub_rows  = cand_rows[mask_proc]
sub_cols  = cand_cols[mask_proc]
sub_dist  = cand_dist[mask_proc]
sub_ang   = cand_ang[mask_proc]

for r, c, d, ang in zip(sub_rows, sub_cols, sub_dist, sub_ang):
    if d == 0:
        visible_local[r, c] = 1
        continue
    s_i = (dem[r, c] - obs_elev) / d
    if s_i >= max_slope[ang] - EPS:       # 可见
        visible_local[r, c] = 1
        if s_i > max_slope[ang]:
            max_slope[ang] = s_i

# ---------- 4. Reduce 可见结果到 rank 0 ----------
visible_int = visible_local
if rank == 0:
    visible_global = np.zeros_like(visible_int, dtype="uint8")
else:
    visible_global = None

comm.Reduce(visible_int, visible_global, op=MPI.MAX, root=0)

# ---------- 5. 绘图 (rank 0) ----------
if rank == 0:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches

    visible = visible_global.astype(bool)
    z_min, z_max = dem.min(), dem.max()
    z_scaled = (dem - z_min) / (z_max - z_min) if z_max > z_min else 0.5
    gray = (z_scaled * 255).astype("uint8")

    green_cmap = ListedColormap([[0, 1, 0, 0], [0, 1, 0, 0.4]])
    extent = (left, right, bottom, top)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(gray, cmap="gray", origin="upper", extent=extent)
    ax.imshow(visible, cmap=green_cmap, origin="upper", extent=extent,
              interpolation="nearest")

    # 纤细红点 & 橙圈
    ax.scatter(obs_x, obs_y, c="red", marker="x",
               s=30, linewidths=0.7, zorder=3)
    circle = mpatches.Circle((obs_x, obs_y), R, fill=False,
                             edgecolor="orange", linewidth=0.7,
                             linestyle="--", alpha=0.6, zorder=2)
    ax.add_patch(circle)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("MPI Viewshed (green = visible, orange = R)")
    ax.legend(["Observer", "Scan radius"], loc="upper right",
              frameon=False, fontsize="small")
    fig.savefig("viewshed_mpi.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("✅ 结果已保存为 viewshed_mpi.png")
