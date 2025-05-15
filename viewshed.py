from pathlib import Path
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import array_bounds, rowcol
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

EPS = 1e-9           # 浮点容差，保证坡度“相等”也算可见

# ---------- 0. 读取 & 拼接所有 tif（nodata → 0） ----------
data_dir = Path("./data")
src_files = [rasterio.open(p) for p in sorted(data_dir.glob("*.tif"))]
if not src_files:
    raise FileNotFoundError("未在 ./data 找到任何 .tif 文件")

mosaic_masked, trans = merge(src_files, masked=True)
dem = mosaic_masked[0].filled(0).astype("float32")   # (rows, cols)

rows, cols = dem.shape
left, bottom, right, top = array_bounds(rows, cols, trans)

# ---------- 1. 打印坐标范围 + 粒度 ----------
dx = abs(trans.a)
dy = abs(trans.e)
print(f"可用坐标范围：X ∈ [{left:.2f}, {right:.2f}],  Y ∈ [{bottom:.2f}, {top:.2f}]")
print(f"坐标可精确到像元粒度：ΔX ≈ {dx:.3f} , ΔY ≈ {dy:.3f}")
print("示例输入：", f"{(left+right)/2:.2f}", ",", f"{(bottom+top)/2:.2f}")

# ---------- 2. 输入观察点 & 半径 ----------
obs_x = float(input("请输入观察者 X 坐标: "))
obs_y = float(input("请输入观察者 Y 坐标: "))
radius = float(input("请输入扫描半径 R: "))

# ---------- 3. 坐标 → 行列 ----------
obs_row, obs_col = rowcol(trans, obs_x, obs_y)
if not (0 <= obs_row < rows and 0 <= obs_col < cols):
    raise ValueError("观察者坐标超出 DEM 范围")
obs_elev = dem[obs_row, obs_col]

# ---------- 4. 预计算距离 ----------
rr, cc = np.indices((rows, cols))
dx_pix = (cc - obs_col) * trans.a
dy_pix = (rr - obs_row) * -trans.e
dist = np.hypot(dx_pix, dy_pix)
within = dist <= radius

# ---------- 5. 视域分析 ----------
visible = np.zeros_like(dem, dtype=bool)
visible[obs_row, obs_col] = True
candidates = np.column_stack(np.where(within))
order = np.argsort(dist[within])
candidates = candidates[order]
max_slope = {}

for r, c in candidates[1:]:
    dx_i = (c - obs_col) * trans.a
    dy_i = (r - obs_row) * -trans.e
    d = np.hypot(dx_i, dy_i)
    if d == 0:
        continue
    s_i = (dem[r, c] - obs_elev) / d
    angle = int(np.degrees(np.arctan2(dy_i, dx_i)) * 2)   # 0.5° 桶
    prev = max_slope.get(angle, -np.inf)
    if s_i < prev - EPS:          # 明显更低 → 被挡住
        continue
    visible[r, c] = True          # 可见（坡度 >= prev - EPS）
    if s_i > prev:                # 只有更高才刷新最大坡度
        max_slope[angle] = s_i

# ---------- 6. 绘图 ----------
z_min, z_max = dem.min(), dem.max()
z_scaled = (dem - z_min) / (z_max - z_min) if z_max > z_min else 0.5
gray = (z_scaled * 255).astype("uint8")

green_cmap = ListedColormap([[0, 1, 0, 0], [0, 1, 0, 0.4]])
extent = (left, right, bottom, top)

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(gray, cmap="gray", origin="upper", extent=extent)
ax.imshow(visible, cmap=green_cmap, origin="upper", extent=extent,
          interpolation="nearest")

# —— ⬇︎ 细化符号与圆 —— #
ax.scatter(obs_x, obs_y,
           c="red", marker="x", s=30, linewidths=0.4, zorder=3,
           label="Observer")

circle = mpatches.Circle((obs_x, obs_y), radius,
                         fill=False, edgecolor="orange",
                         linewidth=0.4, linestyle="--", alpha=0.6,
                         zorder=2, label="Scan radius")
ax.add_patch(circle)
# —— ⬆︎ ———————————— #

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Viewshed (green = visible, orange = R)")
ax.legend(loc="upper right", frameon=False, fontsize="small")

fig.savefig("viewshed.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

print("已保存 viewshed.png")
