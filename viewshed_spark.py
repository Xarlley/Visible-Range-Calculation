#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apache Spark 视域批处理
每行 <x y r> 生成一张 viewshed PNG
用法:
spark-submit viewshed_spark.py \
    --dem_dir ./data \
    --obs_file observers.txt \
    --out_dir ./spark_out
"""
import argparse, os, math, io
from pathlib import Path
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import rowcol, array_bounds
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
from pyspark.sql import SparkSession, functions as F, types as T

# ------------------------ 参数 ------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dem_dir", required=True,
                    help="包含若干 tif 的目录")
parser.add_argument("--obs_file", required=True,
                    help="文本文件，每行 x y r")
parser.add_argument("--out_dir", required=True,
                    help="输出图片目录")
parser.add_argument("--angle_res", type=float, default=0.5,
                    help="角度桶分辨率(°)，默认0.5")
parser.add_argument("--eps", type=float, default=1e-9,
                    help="浮点容差")
args = parser.parse_args()

# ------------------------ Spark Init -----------------
spark = SparkSession.builder.appName("ViewshedSpark").getOrCreate()
sc = spark.sparkContext

# ------------------------ 0. 读取 DEM ----------------
tifs = sorted(Path(args.dem_dir).glob("*.tif"))
if not tifs:
    raise FileNotFoundError("DEM 目录为空")

srcs = [rasterio.open(p) for p in tifs]
mosaic_masked, transform = merge(srcs, masked=True)
dem = mosaic_masked[0].filled(0).astype("float32")
for s in srcs:
    s.close()

# 基本参数
rows, cols = dem.shape
angle_res = args.angle_res
n_bucket = int(360 / angle_res)

# 广播
bc_dem = sc.broadcast(dem)
bc_trans = sc.broadcast(transform)
bc_rows = sc.broadcast(rows)
bc_cols = sc.broadcast(cols)
bc_angle_res = sc.broadcast(angle_res)
bc_n_bucket = sc.broadcast(n_bucket)
bc_eps = sc.broadcast(args.eps)

# ---------------- 1. 读取 observers 文件 --------------
schema = T.StructType([
    T.StructField("x", T.DoubleType(), False),
    T.StructField("y", T.DoubleType(), False),
    T.StructField("r", T.DoubleType(), False)
])
obs_df = spark.read.csv(args.obs_file, sep=r"\s+", schema=schema)

# 给每行加 index 作为文件名后缀
obs_df = obs_df.withColumn("idx", F.monotonically_increasing_id())

# ---------------- 2. 核心计算函数 --------------------
def compute_viewshed(partition):
    import numpy as np, rasterio
    from rasterio.transform import rowcol
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Circle
    angle_res = bc_angle_res.value
    n_bucket  = bc_n_bucket.value
    EPS       = bc_eps.value
    dem = bc_dem.value
    transform = bc_trans.value
    rows = bc_rows.value
    cols = bc_cols.value

    rr_full, cc_full = np.indices((rows, cols))
    dx_pix_full = rr_full * 0  # 占位，后面局部使用避免重复 alloc

    # 视域函数 (单点)
    def viewshed_single(x0, y0, R):
        obs_row, obs_col = rowcol(transform, x0, y0)
        if not (0 <= obs_row < rows and 0 <= obs_col < cols):
            return None
        obs_elev = dem[obs_row, obs_col]

        # 距离矩阵 (复用全局 idx)
        dx = (cc_full - obs_col) * transform.a
        dy = (rr_full - obs_row) * -transform.e
        dist = np.hypot(dx, dy)
        within = dist <= R

        angles_idx = (((np.degrees(np.arctan2(dy, dx)) + 360) % 360)
                      / angle_res).astype("int16")
        cand_rows = rr_full[within]
        cand_cols = cc_full[within]
        cand_dist = dist[within]
        cand_ang  = angles_idx[within]

        order = np.argsort(cand_dist)
        cand_rows = cand_rows[order]
        cand_cols = cand_cols[order]
        cand_dist = cand_dist[order]
        cand_ang  = cand_ang[order]

        visible = np.zeros_like(dist, dtype="uint8")
        max_slope = np.full(n_bucket, -np.inf, dtype="float32")

        for r, c, d, ang in zip(cand_rows, cand_cols, cand_dist, cand_ang):
            if d == 0:
                visible[r, c] = 1
                continue
            s_i = (dem[r, c] - obs_elev) / d
            if s_i >= max_slope[ang] - EPS:
                visible[r, c] = 1
                if s_i > max_slope[ang]:
                    max_slope[ang] = s_i
        return visible.astype(bool)

    # 绘图函数
    def render_png(visible, x0, y0, R, idx):
        z_min, z_max = dem.min(), dem.max()
        z_scaled = (dem - z_min) / (z_max - z_min) if z_max > z_min else 0.5
        gray = (z_scaled * 255).astype("uint8")

        extent = array_bounds(rows, cols, transform)
        green_cmap = ListedColormap([[0,1,0,0], [0,1,0,0.4]])

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(gray, cmap="gray", origin="upper", extent=extent)
        ax.imshow(visible, cmap=green_cmap, origin="upper",
                  extent=extent, interpolation="nearest")
        ax.scatter(x0, y0, c="red", marker="x", s=30, linewidths=0.7, zorder=3)
        circle = Circle((x0, y0), R, fill=False, edgecolor="orange",
                        linewidth=0.7, linestyle="--", alpha=0.6, zorder=2)
        ax.add_patch(circle)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Viewshed idx={idx}")
        out_path = os.path.join(args.out_dir, f"viewshed_{idx}.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path

    for row in partition:
        x0, y0, R, idx = row.x, row.y, row.r, int(row.idx)
        vis = viewshed_single(x0, y0, R)
        if vis is None:
            print(f"[idx {idx}] 坐标超界或无数据，跳过")
            continue
        path = render_png(vis, x0, y0, R, idx)
        print(f"[idx {idx}] 输出 -> {path}")
    yield from ()  # 无需返回 RDD 结果

# ---------------- 3. 触发计算 -------------------------
Path(args.out_dir).mkdir(parents=True, exist_ok=True)
(obs_df.repartition(spark.sparkContext.defaultParallelism)
      .rdd
      .mapPartitions(compute_viewshed)
      .count())   # action, 强制执行

spark.stop()
print("✅ 全部任务完成，图片已写入", args.out_dir)
