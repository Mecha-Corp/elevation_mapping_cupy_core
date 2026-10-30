"""Exercise the optional ANYbotics fusion through the native map owner."""

from pathlib import Path

import cupy as cp
import numpy as np
import pytest

from elevation_mapping_cupy import ElevationMap, Parameter


@pytest.fixture
def terrain():
    param = Parameter(
        resolution=0.1, map_length=4.0, weight_file="", use_chainer=False,
        plugin_config_file=str(Path(__file__).parents[1] / "configs/plugin_config.yaml"),
        subscriber_cfg={}, enable_multi_height_fusion=True,
        enable_visibility_cleanup=False, enable_drift_compensation=False,
        enable_overlap_clearance=False,
    )
    param.update()
    return ElevationMap(param)


def cell(terrain):
    return terrain.elevation_map[:, terrain.cell_n // 2 + 10, terrain.cell_n // 2]


def feed(terrain, heights):
    points = np.array([[1.05, 0.05, h] for h in heights], dtype=np.float32)
    terrain.input_pointcloud(points, ["x", "y", "z"], np.eye(3), np.zeros(3), 0, 0)
    return cp.asnumpy(cell(terrain))


@pytest.mark.parametrize("height,expected", [(0.0, 0.127), (0.2, 0.2)])
def test_recent_surface_rejects_lower_outlier_and_accepts_higher(terrain, height, expected):
    cell(terrain)[0:3] = cp.array([0.127, 0.0001, 1])
    cell(terrain)[4] = 0.1
    result = feed(terrain, [height])
    assert result[0] == pytest.approx(expected)
    assert result[4] == pytest.approx(0.1)  # Outliers do not refresh the clock.


def test_old_surface_increases_uncertainty_without_replacing_height(terrain):
    cell(terrain)[0:3] = cp.array([0.0, 0.0001, 1])
    terrain.param.time_interval = 2.0
    terrain.update_time()
    result = feed(terrain, [0.127])
    assert result[0] == 0
    assert result[1] == pytest.approx(0.0001 + terrain.param.multi_height_noise)
    assert result[4] == 2


def test_inliers_accumulate_information_sequentially(terrain):
    variance = terrain.param.sensor_noise_factor * (1.05 ** 2 + 0.05 ** 2)
    result = feed(terrain, [0.0] * 10)
    assert result[0] == 0
    assert result[1] == pytest.approx(variance / 10, rel=1e-5)
    assert result[4] == 0


def test_first_measurement_does_not_blend_with_unobserved_zero(terrain):
    result = feed(terrain, [0.127])
    assert result[0] == pytest.approx(0.127)
    assert result[2] == 1
    assert result[4] == 0  # Native observed cells must publish a finite age.


def test_point_order_is_preserved_inside_a_cell(terrain):
    cell(terrain)[0:3] = cp.array([0.127, 0.0001, 1])
    cell(terrain)[4] = 0
    # Lower outlier is ignored; higher outlier replaces the surface.
    result = feed(terrain, [0.0, 0.2])
    assert result[0] == pytest.approx(0.2)
    # In the reverse order, the high return first raises the cell variance,
    # so the low return is an inlier. This order dependence is upstream's.
    cell(terrain)[0:3] = cp.array([0.127, 0.0001, 1])
    result = feed(terrain, [0.2, 0.0])
    assert 0 < result[0] < 0.2
