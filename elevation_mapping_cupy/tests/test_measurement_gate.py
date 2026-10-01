"""Exercise the CUDA fusion gate with sparse noisy floor and stair returns."""

import cupy as cp
import numpy as np

from elevation_mapping_cupy.elevation_mapping import ElevationMap
from elevation_mapping_cupy.parameter import Parameter


def test_sparse_noisy_steps_remain_consistent_and_reject_height_outlier(tmp_path):
    plugins = tmp_path / 'plugins.yaml'
    plugins.write_text('{}\n')
    param = Parameter(
        resolution=0.025, map_length=2.0, sensor_noise_factor=0.001,
        min_valid_distance=0.05,
        subscriber_cfg={}, additional_layers=[], fusion_algorithms=[],
        weight_file='', plugin_config_file=str(plugins),
        enable_drift_compensation=False, enable_visibility_cleanup=False,
    )
    param.update()
    mapper = ElevationMap(param)
    x, y = np.meshgrid(np.arange(-.4, .4, .05), np.arange(-.4, .4, .05))
    surface = np.where(x >= 0, .1, 0.)
    cloud = np.column_stack((x.ravel(), y.ravel(), surface.ravel() - .5))

    def fuse(noise):
        points = cloud.copy()
        points[:, 2] += noise
        mapper.input_pointcloud(points, ['x', 'y', 'z'], np.eye(3), np.array([0., 0., .5]), 0., 0.)
        height = cp.asnumpy(mapper.get_map_with_name_ref('elevation', return_cupy=True))
        variance = cp.asnumpy(mapper.get_map_with_name_ref('variance', return_cupy=True))
        return height, variance

    # Each cell has one point per scan. A 6 mm inter-scan change is consistent
    # with this range-noise prior; comparing metres with variance rejects it.
    fuse(-.003)
    height, variance = fuse(.003)
    observed = np.isfinite(height)
    assert observed.sum() == len(cloud)
    assert (variance[observed] < .001).all()
    assert (np.minimum(abs(height[observed]), abs(height[observed] - .1)) < .004).all()
    # A 30 cm jump is inconsistent: it must not silently replace the two steps.
    changed, variance = fuse(.3)
    np.testing.assert_array_equal(changed[observed], height[observed])
    assert (variance[observed] >= .01).all()
