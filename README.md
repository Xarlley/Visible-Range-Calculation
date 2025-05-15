# Visible-Range-Calculation

## Environment

### normal version

For `tif2png.py`, `viewshed.py`, `debug_dem_query.py`, `debug_circle_max.py`.

```bash
pip install mpi4py rasterio numpy matplotlib
```

### mpi version

For `viewshed_mpi.py`.

On ubuntu,

```bash
apt-get install -y openmpi-bin libopenmpi-dev
```

On centos,

```bash
dnf install openmpi-devel
```

```bash
pip install mpi4py rasterio numpy matplotlib
```

### spark version

For `viewshed_spark.py`.

```bash
pip install pyspark rasterio numpy matplotlib
```

Install the environment on all executor nodes.

## How to run

Put `*.tif` file into `./data`. This program can scan them all and merge them into **one** image.

### Generate base image

```bash
python tif2png.py
```

### Generate visible range image

```bash
python viewshed.py
```

### Generate visible range image with mpi

```bash
mpiexec -n 8 python viewshed_mpi.py
```

### Generate visible range image with spark

```bash
spark-submit --master local[8] viewshed_spark.py \
             --dem_dir ./data \
             --obs_file observers.txt \
             --out_dir ./spark_out
```

`observers.txt` should be organized as,

```
x y r
```

such as,

```
112 22 1
113 23 0.5
112.55 24.32 0.5
```