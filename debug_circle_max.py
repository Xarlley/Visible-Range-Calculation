#!/usr/bin/env python3
"""
debug_circle_max.py
连续读取用户输入 (x y R)，返回该圆形区域中的最高点坐标与高程
"""

from pathlib import Path
import numpy as np
import rasterio
import math
from rasterio.merge import merge
from rasterio.transform import rowcol, xy, array_bounds

DATA_DIR = Path("./data")

def load_dem():
    """拼接 ./data/*.tif → MaskedArray dem"""
    tifs = sorted(DATA_DIR.glob("*.tif"))
    if not tifs:
        raise FileNotFoundError("未在 ./data 找到任何 .tif 文件")
    srcs = [rasterio.open(p) for p in tifs]
    mosaic_masked, transform = merge(srcs, masked=True)
    dem = mosaic_masked[0]                 # MaskedArray
    bounds = array_bounds(*dem.shape, transform)
    res_x, res_y = abs(transform.a), abs(transform.e)
    for s in srcs:
        s.close()
    return dem, transform, bounds, res_x, res_y

def circle_max(dem, transform, cx, cy, R):
    """返回圆内最高点 (x, y, z); 若无有效像元则返回 None"""
    rows, cols = dem.shape
    center_row, center_col = rowcol(transform, cx, cy)
    # 粗略窗口
    pix_r = int(np.ceil(R / abs(transform.a)))   # 使用 X 分辨率近似
    r0, r1 = max(0, center_row - pix_r), min(rows, center_row + pix_r + 1)
    c0, c1 = max(0, center_col - pix_r), min(cols, center_col + pix_r + 1)

    if r0 >= r1 or c0 >= c1:
        return None

    sub = dem[r0:r1, c0:c1]          # MaskedArray (可能全部 masked)
    if sub.mask.all():
        return None

    # 准确距离筛选
    rr, cc = np.indices(sub.shape)
    rr += r0
    cc += c0
    dx = (cc - center_col) * transform.a
    dy = (rr - center_row) * -transform.e
    dist = np.hypot(dx, dy)

    within = dist <= R
    if not within.any():
        return None

    # Combined mask：圆外或 nodata
    mask = ~within | sub.mask
    if mask.all():
        return None

    sub_valid = np.ma.MaskedArray(sub.data, mask)
    z_max = sub_valid.max()                   # 忽略掩膜
    if sub_valid.mask.all():
        return None

    # 找到 z_max 位置（取第一个即可）
    pos = np.ma.where(sub_valid == z_max)
    r_max, c_max = int(pos[0][0]) + r0, int(pos[1][0]) + c0
    x_max, y_max = xy(transform, r_max, c_max, offset="center")
    return x_max, y_max, float(z_max)

def main():
    dem, transform, (left, bot, right, top), res_x, res_y = load_dem()
    print("=" * 60)
    print("DEM 已加载完毕")
    print(f"可用坐标范围：X ∈ [{left:.3f}, {right:.3f}],  Y ∈ [{bot:.3f}, {top:.3f}]")
    print(f"坐标粒度：ΔX ≈ {res_x:.3f} , ΔY ≈ {res_y:.3f}")
    print("输入格式：x y R   （空格分隔）")
    print("退出：exit / quit / q")
    print("=" * 60)

    while True:
        raw = input("> ").strip()
        if raw.lower() in {"exit", "quit", "q"}:
            break
        try:
            x_str, y_str, r_str = raw.split()
            x0, y0, R = float(x_str), float(y_str), float(r_str)
        except ValueError:
            print("❌ 请按格式输入三个数字：x y R")
            continue
        if not (left <= x0 <= right and bot <= y0 <= top):
            print("⚠️  坐标超出 DEM 范围")
            continue
        if R <= 0:
            print("⚠️  半径 R 必须为正")
            continue

        result = circle_max(dem, transform, x0, y0, R)
        if result is None:
            print("ℹ️  圆内无有效 DEM 数据")
        else:
            x_max, y_max, z_max = result
            print(f"✅ 最高点：({x_max:.3f}, {y_max:.3f})   高程 = {z_max}")

    print("程序结束，再见！")

if __name__ == "__main__":
    main()
