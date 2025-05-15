#!/usr/bin/env python3
"""
debug_dem_query.py
读取 ./data 目录内所有 GeoTIFF，合成为单幅 DEM，
随后循环读取用户输入的坐标并输出对应高程。
输入格式：x y
退出：    exit / quit / q
"""

from pathlib import Path
import rasterio
from rasterio.merge import merge
from rasterio.transform import rowcol, array_bounds
import numpy as np

DATA_DIR = Path("./data")        # 修改路径请改这里

def load_mosaic():
    """拼接 ./data/*.tif → (dem, transform, bounds)"""
    tif_paths = sorted(DATA_DIR.glob("*.tif"))
    if not tif_paths:
        raise FileNotFoundError("未在 ./data 找到任何 .tif 文件")
    srcs = [rasterio.open(p) for p in tif_paths]
    mosaic, transform = merge(srcs, masked=True)  # mosaic.shape = (1, rows, cols)
    dem = mosaic[0].astype("float32")             # 提取单波段并转 float32
    bounds = array_bounds(dem.shape[0], dem.shape[1], transform)
    for s in srcs:                                # 释放文件句柄
        s.close()
    return dem, transform, bounds

def main():
    dem, transform, (left, bottom, right, top) = load_mosaic()
    res_x = abs(transform.a)
    res_y = abs(transform.e)          # e 通常为负，取绝对值
    print("=" * 60)
    print("DEM 已加载完毕")
    print(f"可用坐标范围：X ∈ [{left:.3f}, {right:.3f}],  Y ∈ [{bottom:.3f}, {top:.3f}]")
    print(f"坐标可精确到的粒度：ΔX ≈ {res_x:.3f} , ΔY ≈ {res_y:.3f}")
    print("请输入坐标（格式：x y），或输入 exit/quit/q 退出")
    print("=" * 60)

    while True:
        raw = input("> ").strip()
        if raw.lower() in {"exit", "quit", "q"}:
            break
        try:
            x_str, y_str = raw.split()
            x, y = float(x_str), float(y_str)
        except ValueError:
            print("❌ 请输入两个数字，用空格分隔，例如：415123.5 3220456.0")
            continue

        if not (left <= x <= right and bottom <= y <= top):
            print("⚠️  坐标超出 DEM 范围")
            continue

        # rowcol 返回 (row, col) 整数索引
        row, col = rowcol(transform, x, y)
        if not (0 <= row < dem.shape[0] and 0 <= col < dem.shape[1]):
            print("⚠️  坐标映射后超出数组范围")
            continue

        z = dem[row, col]
        if np.isnan(z):
            print("ℹ️  该位置为 NoData")
        else:
            print(f"✅ 高程：{z}")

    print("程序结束，再见！")

if __name__ == "__main__":
    main()
