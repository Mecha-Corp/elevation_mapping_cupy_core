# elevation_mapping_cupy_core

A standalone, ROS-free Python package extracted from [elevation_mapping_cupy](https://github.com/leggedrobotics/elevation_mapping_cupy) (ETH Zurich / Takahiro Miki).

The upstream project is a full ROS2 node for real-time GPU-accelerated elevation mapping on legged robots. This package strips all ROS dependencies and exposes only the core Python library, making it usable in:

- **Simulation** pipelines (Isaac Sim, Gazebo, custom envs)
- **Inference** pipelines that consume elevation or semantic maps
- Any **standalone Python** code that needs GPU-backed elevation mapping without a ROS runtime

## What's included

- `ElevationMap` — core CuPy-based elevation map with GPU kernels for point cloud fusion, traversability filtering, and map shifting
- `SemanticMap` — semantic layer management with pluggable fusion strategies (Bayesian, class-average, class-max, etc.)
- `PluginManager` — post-processing plugins (inpainting, erosion, smooth filter, traversability, PCA features, etc.)
- `Parameter` — unified configuration dataclass
- Custom CUDA kernels for fast point cloud ingestion and map updates

## Installation

```bash
pip install elevation-mapping-cupy-core
```

Requires a CUDA 12.x environment with CuPy installed (`cupy-cuda12x`).

## Quick start

```python
from elevation_mapping_cupy import ElevationMap, Parameter

param = Parameter(resolution=0.04, map_length=8.0)
em = ElevationMap(param)

# Feed a point cloud (N, 3) numpy array
em.input(points, R, t, position_noise, orientation_noise)

# Retrieve the elevation layer as a numpy array
elevation = em.get_map_with_name("elevation")
```

## License

MIT — Copyright (c) 2022 ETH Zurich, Takahiro Miki. See [LICENSE](LICENSE).
The optional ANYbotics multi-height fusion port carries the
[BSD-3-Clause notice](LICENSE-ANYbotics.txt).

## Sequential multi-height fusion

`Parameter(enable_multi_height_fusion=True)` selects the ANYbotics per-cell
update in [multi_height.py](elevation_mapping_cupy/kernels/multi_height.py).
It replaces averaged point fusion and density-based edge sharpening; existing
map movement, visibility cleanup, drift compensation and publication still
belong to `ElevationMap`. The default remains the original CuPy fusion.

The caller must advance `update_time()` using acquisition-time intervals, as
with the original mapper. New parameter defaults and upstream provenance are
declared in `Parameter`; the port's intentional differences are documented at
the kernel. This option does not select a sensor noise model or sensor preset.
