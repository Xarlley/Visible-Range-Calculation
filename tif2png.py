from pathlib import Path
import rasterio
from rasterio.merge import merge
from rasterio.transform import array_bounds
import numpy as np
import matplotlib.pyplot as plt

data_dir = Path("./data")
tif_paths = sorted(data_dir.glob("*.tif"))
if not tif_paths:
    raise FileNotFoundError("./data 文件夹中未找到任何 *.tif")

# ---------- 1. 打开所有 tif ----------
src_files = [rasterio.open(p) for p in tif_paths]

# ---------- 2. 虚拟镶嵌为单幅影像 ----------
mosaic, out_trans = merge(src_files, masked=True)  # masked=True → nodata 保留掩膜
z = mosaic[0].astype("float32")                    # (rows, cols)

# ---------- 3. 全局高程范围 ----------
finite_mask = np.isfinite(z)
if not finite_mask.any():
    raise ValueError("所有影像均为 nodata！")
z_min, z_max = np.nanmin(z), np.nanmax(z)

# ---------- 4. 归一化 → 灰度 ----------
if z_max > z_min:
    z_scaled = (z - z_min) / (z_max - z_min)
else:
    z_scaled = np.full_like(z, 0.5, dtype="float32")   # 常数影像
z_scaled = np.nan_to_num(z_scaled, nan=0.0)            # nodata → 黑
img = (z_scaled * 255).astype("uint8")

# ---------- 5. 计算外包框 → 绘图 ----------
h, w = z.shape
left, bottom, right, top = array_bounds(h, w, out_trans)

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(img, cmap="gray", origin="upper",
          extent=(left, right, bottom, top))
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Merged Elevation (higher = whiter)")

out_png = "mosaic_elevation.png"
fig.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

print(f"已生成 {out_png}")
