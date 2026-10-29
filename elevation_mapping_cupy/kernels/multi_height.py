# Adapted from elevation_mapping, Copyright 2019, ANYbotics AG.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Sequential per-cell fusion from ANYbotics ElevationMap::add.

https://github.com/ANYbotics/elevation_mapping/blob/f4b082c64a3e660980da53b33c7936a8f2a2ea22/elevation_mapping/src/ElevationMap.cpp#L128-L177

Distinct cells run in parallel; points in one cell retain cloud order. Recent
lower outliers are ignored, higher ones replace/blend the surface, and older
outliers increase uncertainty. Inliers use sequential Kalman updates, rather
than averaging posteriors computed against one shared prior. This mode replaces
the density-based edge rule, not the sensor model, drift correction or cleanup.

The existing time layer stores age, equivalent to upstream's scan time minus
last update time. Only an ordinary inlier refreshes an existing cell, as upstream
does. One intentional difference: the initial measurement also initializes age
to zero. Upstream leaves its time undefined until the next inlier; the native
CuPy/host contract requires every observed cell to have a finite age.
Height, variance, validity and age stay owned by ElevationMap and roll together.
CuPy's upper-bound and semantic count layers are maintained for its consumers;
ROS, horizontal covariance and color fusion are not imported.
"""

import string

import cupy as cp

from .custom_kernels import map_utils


def multi_height_kernels(param):
    """Build point grouping and per-cell fusion kernels using native geometry."""
    preamble = map_utils(
        param.resolution, param.cell_n, param.cell_n, param.sensor_noise_factor,
        param.min_valid_distance, param.max_height_range,
        param.ramped_height_range_a, param.ramped_height_range_b,
        param.ramped_height_range_c,
    )
    index = cp.ElementwiseKernel(
        "raw T p, raw T R, raw T t",
        "int64 index, T z, T variance",
        preamble=preamble,
        operation=string.Template("""
            T rx = p[i * 3], ry = p[i * 3 + 1], rz = p[i * 3 + 2];
            T x = transform_p(rx, ry, rz, R[0], R[1], R[2], t[0]);
            T y = transform_p(rx, ry, rz, R[3], R[4], R[5], t[1]);
            z = transform_p(rx, ry, rz, R[6], R[7], R[8], t[2]);
            variance = point_noise(rx, ry, rz);
            index = get_idx(x, y, 0, 0);
            if (!is_inside(index) || !is_valid(x, y, z, t[0], t[1], t[2])) {
                index = ${cells};  // Sorted beyond all real cells.
            }
        """).substitute(cells=param.cell_n ** 2),
        name="multi_height_index",
    )
    fusion = cp.ElementwiseKernel(
        "raw int64 order, raw int64 offsets, raw T heights, raw T variances",
        "raw T map, raw T newmap",
        preamble="#include <math_constants.h>\n" + preamble,
        operation=string.Template("""
            T h = map[get_map_idx(i, 0)];
            T v = map[get_map_idx(i, 1)];
            bool valid = map[get_map_idx(i, 2)] > 0.5;
            T age = map[get_map_idx(i, 4)];
            int accepted = 0;
            for (long long j = offsets[i]; j < offsets[i + 1]; ++j) {
                long long k = order[j];
                T z = heights[k], pv = variances[k];
                if (!valid) {
                    h = z;
                    v = pv;
                    valid = true;
                    age = 0;
                    ++accepted;
                    continue;
                }
                // Upstream deliberately gates on cell variance, not v + pv.
                if (fabs(z - h) > ${mahalanobis} * sqrt(v)) {
                    if (age <= ${duration}) {
                        if (z > h) {
                            h = ${alpha} * h + (1.0 - ${alpha}) * z;
                            v = ${alpha} * v + (1.0 - ${alpha}) * pv;
                            ++accepted;
                        }
                    } else {
                        v += ${multi_height_noise};
                    }
                    continue;
                }
                h = (v * z + pv * h) / (v + pv);
                v = (pv * v) / (v + pv);
                age = 0;
                ++accepted;
            }
            if (valid) {
                // ElevationMap::clean bounds variance once per cloud.
                v = v < ${min_variance} ? ${min_variance}
                    : (v > ${max_variance} ? CUDART_INF_F : v);
                map[get_map_idx(i, 0)] = h;
                map[get_map_idx(i, 1)] = v;
                map[get_map_idx(i, 2)] = isfinite(v) ? 1 : 0;
                map[get_map_idx(i, 4)] = age;
                if (accepted > 0) {
                    map[get_map_idx(i, 5)] = h;
                    map[get_map_idx(i, 6)] = 0;
                    newmap[get_map_idx(i, 0)] = h * accepted;
                    newmap[get_map_idx(i, 1)] = v * accepted;
                    newmap[get_map_idx(i, 2)] = accepted;
                }
            }
        """).substitute(
            mahalanobis=param.mahalanobis_thresh,
            duration=param.scanning_duration,
            alpha=param.increase_height_alpha,
            multi_height_noise=param.multi_height_noise,
            min_variance=param.min_variance,
            max_variance=param.max_variance,
        ),
        name="multi_height_fusion",
    )
    return index, fusion
