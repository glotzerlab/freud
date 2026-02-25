# Copyright (c) 2010-2026 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pytest
import util
from test_managedarray import ManagedArrayTestBase

import freud

matplotlib.use("agg")


class TestCorrelationFunction:
    def test_generate_bins(self):
        r_max = 5
        dr = 0.1
        bins = round(r_max / dr)

        ocf = freud.density.CorrelationFunction(bins, r_max)

        # make sure the radius for each bin is generated correctly
        r_list = np.array(
            [dr * (i + 1 / 2) for i in range(bins) if dr * (i + 1 / 2) < r_max]
        )
        npt.assert_allclose(ocf.bin_centers, r_list, rtol=1e-4, atol=1e-4)
        npt.assert_allclose((ocf.bin_edges + dr / 2)[:-1], r_list, rtol=1e-4, atol=1e-4)

    def test_attribute_access(self):
        r_max = 10.0
        bins = 10
        num_points = 100
        box_size = r_max * 3.1
        box, points = freud.data.make_random_system(
            box_size, num_points, is2D=True, seed=1
        )
        ang = np.random.random_sample(num_points).astype(np.float64) * 2.0 * np.pi
        ocf = freud.density.CorrelationFunction(bins, r_max)

        # Test protected attribute access
        with pytest.raises(AttributeError):
            ocf.correlation
        with pytest.raises(AttributeError):
            ocf.box
        with pytest.raises(AttributeError):
            ocf.bin_counts

        ocf.compute((box, points), ang, reset=False)

        # Test if accessible now
        ocf.correlation
        ocf.box
        ocf.bin_counts

        ocf.compute((box, points), ang)

        # Test if accessible now
        ocf.correlation
        ocf.box
        ocf.bin_counts

    def test_random_points_complex(self):
        r_max = 10.0
        bins = 10
        num_points = 1000
        box_size = r_max * 3.1
        box, points = freud.data.make_random_system(
            box_size, num_points, is2D=True, seed=1
        )
        ang = np.random.random_sample(num_points).astype(np.float64) * 2.0 * np.pi
        comp = np.exp(1j * ang)
        expected = np.zeros(bins, dtype=np.complex64)
        absolute_tolerance = 0.1
        # first bin is bad
        test_set = util.make_raw_query_nlist_test_set(
            box, points, points, "ball", r_max, 0, True
        )
        for nq, neighbors in test_set:
            ocf = freud.density.CorrelationFunction(bins, r_max)
            ocf.compute(nq, comp, neighbors=neighbors)
            npt.assert_allclose(ocf.correlation, expected, atol=absolute_tolerance)
            assert box == ocf.box

            # Test setting that the reset flag works as expected.
            ocf.compute(nq, comp, neighbors=neighbors, reset=False)
            npt.assert_allclose(ocf.correlation, expected, atol=absolute_tolerance)
            ocf.compute(nq, comp, neighbors=neighbors)
            npt.assert_allclose(ocf.correlation, expected, atol=absolute_tolerance)

    def test_random_points_real(self):
        r_max = 10.0
        bins = 10
        num_points = 1000
        box_size = r_max * 3.1
        box, points = freud.data.make_random_system(
            box_size, num_points, is2D=True, seed=1
        )
        ang = np.random.random_sample(num_points).astype(np.float64) - 0.5
        expected = np.zeros(bins, dtype=np.float64)
        absolute_tolerance = 0.1
        # first bin is bad
        test_set = util.make_raw_query_nlist_test_set(
            box, points, points, "ball", r_max, 0, True
        )
        for nq, neighbors in test_set:
            ocf = freud.density.CorrelationFunction(bins, r_max)
            ocf.compute(nq, ang, neighbors=neighbors, reset=False)
            npt.assert_allclose(ocf.correlation, expected, atol=absolute_tolerance)
            ocf.compute(nq, ang, neighbors=neighbors)
            npt.assert_allclose(ocf.correlation, expected, atol=absolute_tolerance)
            ocf.compute(nq, ang, points, ang, neighbors=neighbors, reset=False)
            npt.assert_allclose(ocf.correlation, expected, atol=absolute_tolerance)
            ocf.compute(nq, ang, query_values=ang, neighbors=neighbors, reset=False)
            npt.assert_allclose(ocf.correlation, expected, atol=absolute_tolerance)
            ocf.compute(nq, ang, neighbors=neighbors)
            npt.assert_allclose(ocf.correlation, expected, atol=absolute_tolerance)
            assert freud.box.Box.square(box_size) == ocf.box

    def test_zero_points_complex(self):
        r_max = 10.0
        bins = 10
        dr = r_max / bins
        num_points = 1000
        box_size = r_max * 3.1
        _box, points = freud.data.make_random_system(
            box_size, num_points, is2D=True, seed=1
        )
        ang = np.zeros(int(num_points), dtype=np.float64)
        comp = np.exp(1j * ang)
        ocf = freud.density.CorrelationFunction(bins, r_max)
        ocf.compute(
            (freud.box.Box.square(box_size), points),
            comp,
            neighbors={"r_max": r_max, "exclude_ii": True},
        )

        expected = np.ones(int(r_max / dr), dtype=np.float64) + 1j * np.zeros(
            int(r_max / dr), dtype=np.float64
        )
        absolute_tolerance = 0.1
        npt.assert_allclose(ocf.correlation, expected, atol=absolute_tolerance)

    def test_zero_points_real(self):
        r_max = 10.0
        dr = 1.0
        num_points = 1000
        box_size = r_max * 3.1
        box, points = freud.data.make_random_system(
            box_size, num_points, is2D=True, seed=1
        )
        ang = np.zeros(int(num_points), dtype=np.float64)
        ocf = freud.density.CorrelationFunction(r_max, dr)
        ocf.compute((box, points), ang)

        expected = np.zeros(int(r_max / dr), dtype=np.float32)
        absolute_tolerance = 0.1
        npt.assert_allclose(ocf.correlation, expected, atol=absolute_tolerance)

    def test_counts(self):
        r_max = 10.0
        bins = 10
        num_points = 10
        box_size = r_max * 2.1
        box, points = freud.data.make_random_system(
            box_size, num_points, is2D=True, seed=1
        )
        ang = np.zeros(int(num_points), dtype=np.float64)
        comp = np.exp(1j * ang)

        vectors = points[np.newaxis, :, :] - points[:, np.newaxis, :]
        vector_lengths = np.array(
            [
                [np.linalg.norm(box.wrap(vectors[i][j])) for i in range(num_points)]
                for j in range(num_points)
            ]
        )

        # Subtract len(points) to exclude the zero i-i distances
        expected = np.sum(vector_lengths < r_max) - len(points)
        ocf = freud.density.CorrelationFunction(bins, r_max)
        ocf.compute(
            (freud.box.Box.square(box_size), points),
            comp,
            neighbors={"r_max": r_max, "exclude_ii": True},
        )
        assert np.sum(ocf.bin_counts) == expected

    @pytest.mark.skip(reason="Skip slow summation test.")
    def test_summation(self):
        # Causes the correlation function to add lots of small numbers together
        # This leads to vastly different results with different numbers of
        # threads if the summation is not done robustly
        N = 20000
        L = 1000
        phi = np.random.rand(N)
        box, pos2d = freud.data.make_random_system(L, N, is2D=True, seed=1)

        # With a small number of particles, we won't get the average exactly
        # right, so we check for different behavior with different numbers of
        # threads
        freud.parallel.setNumThreads(1)
        # A very large bin size exacerbates the problem
        cf = freud.density.CorrelationFunction(L / 2.1, 40)
        cf.compute((box, pos2d), phi)
        c1 = cf.bin_counts
        f1 = np.real(cf.correlation)

        freud.parallel.setNumThreads(20)
        cf.compute((box, pos2d), phi)
        c2 = cf.bin_counts
        f2 = np.real(cf.correlation)

        npt.assert_allclose(f1, f2)
        npt.assert_array_equal(c1, c2)

    def test_repr(self):
        cf = freud.density.CorrelationFunction(1000, 40)
        assert str(cf) == str(eval(repr(cf)))

    def test_repr_png(self):
        r_max = 10.0
        bins = 10
        num_points = 100
        box_size = r_max * 3.1
        _box, points = freud.data.make_random_system(
            box_size, num_points, is2D=True, seed=1
        )
        ang = np.random.random_sample(num_points).astype(np.float64) * 2.0 * np.pi
        comp = np.exp(1j * ang)
        ocf = freud.density.CorrelationFunction(bins, r_max)

        with pytest.raises(AttributeError):
            ocf.plot()
        assert ocf._repr_png_() is None

        ocf.compute(
            (freud.box.Box.square(box_size), points),
            comp,
            neighbors={"r_max": r_max, "exclude_ii": True},
        )
        ocf._repr_png_()
        plt.close("all")

    def test_query_nn_complex(self):
        """Test nearest-neighbor-based querying."""
        box_size = 8
        r_max = 3
        bins = 3
        box = freud.box.Box.cube(box_size)
        points = np.array([[0, 0, 0]], dtype=np.float32)
        query_points = np.array(
            [
                [0.4, 0.0, 0.0],
                [0.9, 0.0, 0.0],
                [0.0, 1.4, 0.0],
                [0.0, 1.9, 0.0],
                [0.0, 0.0, 2.4],
                [0.0, 0.0, 2.9],
            ],
            dtype=np.float32,
        )
        values = np.ones(points.shape[0])
        query_values = np.ones(query_points.shape[0])

        # Normal calculation
        cf = freud.density.CorrelationFunction(bins, r_max)
        cf.compute(
            (box, points),
            values,
            query_points,
            query_values,
            neighbors={"mode": "nearest", "num_neighbors": 1},
        )
        npt.assert_array_equal(cf.correlation, [1, 1, 1])
        npt.assert_array_equal(cf.bin_counts, [2, 2, 2])

        # Swap points and query_points
        cf.compute(
            (box, query_points),
            query_values,
            points,
            values,
            neighbors={"mode": "nearest", "num_neighbors": 1},
        )
        npt.assert_array_equal(cf.correlation, [1, 0, 0])
        npt.assert_array_equal(cf.bin_counts, [1, 0, 0])

        # Make values complex
        values = [1 + 1j]
        query_values = [1 + 1j, 1 + 1j, 2 + 2j, 2 + 2j, 3 + 3j, 3 + 3j]
        cf.compute(
            (box, points),
            values,
            query_points,
            query_values,
            neighbors={"mode": "nearest", "num_neighbors": 1},
        )
        npt.assert_array_equal(cf.correlation, [2, 4, 6])
        npt.assert_array_equal(cf.bin_counts, [2, 2, 2])

        # Test the effect of conjugating the query_values
        cf.compute(
            (box, points),
            np.conj(values),
            query_points,
            query_values,
            neighbors={"mode": "nearest", "num_neighbors": 1},
        )
        npt.assert_array_equal(cf.correlation, [2j, 4j, 6j])
        npt.assert_array_equal(cf.bin_counts, [2, 2, 2])

        # Swap points and query_points
        cf.compute(
            (box, query_points),
            query_values,
            points,
            values,
            neighbors={"mode": "nearest", "num_neighbors": 1},
        )
        npt.assert_array_equal(cf.correlation, [2, 0, 0])
        npt.assert_array_equal(cf.bin_counts, [1, 0, 0])

        # Test the effect of conjugating the query_values
        cf.compute(
            (box, query_points),
            np.conj(query_values),
            points,
            values,
            neighbors={"mode": "nearest", "num_neighbors": 1},
        )
        npt.assert_array_equal(cf.correlation, [2j, 0, 0])
        npt.assert_array_equal(cf.bin_counts, [1, 0, 0])

    def test_query_nn_real(self):
        """Test nearest-neighbor-based querying."""
        box_size = 8
        r_max = 3
        bins = 3
        box = freud.box.Box.cube(box_size)
        points = np.array([[0, 0, 0]], dtype=np.float32)
        query_points = np.array(
            [
                [0.4, 0.0, 0.0],
                [0.9, 0.0, 0.0],
                [0.0, 1.4, 0.0],
                [0.0, 1.9, 0.0],
                [0.0, 0.0, 2.4],
                [0.0, 0.0, 2.9],
            ],
            dtype=np.float32,
        )
        values = np.ones(points.shape[0])
        query_values = np.ones(query_points.shape[0])

        cf = freud.density.CorrelationFunction(bins, r_max)
        cf.compute(
            (box, points),
            values,
            query_points,
            query_values,
            neighbors={"mode": "nearest", "num_neighbors": 1},
        )
        npt.assert_array_equal(cf.correlation, [1, 1, 1])
        npt.assert_array_equal(cf.bin_counts, [2, 2, 2])

        cf.compute(
            (box, query_points),
            query_values,
            points,
            values,
            neighbors={"mode": "nearest", "num_neighbors": 1},
        )
        npt.assert_array_equal(cf.correlation, [1, 0, 0])
        npt.assert_array_equal(cf.bin_counts, [1, 0, 0])

    def test_points_ne_query_points_complex(self):
        # Helper function to give complex number representation of a point
        def value_func(_p):
            return _p[0] + 1j * _p[1]

        r_max = 10.0
        bins = 100
        dr = r_max / bins
        box_size = r_max * 5
        box = freud.box.Box.square(box_size)

        ocf = freud.density.CorrelationFunction(bins, r_max)

        query_points = []
        query_values = []
        N = 300

        # We are essentially generating all n-th roots of unity
        # scalar multiplied by the each bin centers
        # with the value of a point being its complex number representation.
        # Therefore, the correlation should be uniformly zero
        # since the roots of unity add up to zero, if we set our ref_point in
        # the origin.
        # Nice proof for this fact is that when the n-th roots of unity
        # are viewed as vectors, we can draw a regular n-gon
        # so that we start at the origin and come back to origin.
        for r in ocf.bin_centers:
            for k in range(N):
                point = [
                    r * np.cos(2 * np.pi * k / N),
                    r * np.sin(2 * np.pi * k / N),
                    0,
                ]
                query_points.append(point)
                query_values.append(value_func(point))

        expected_correlation = np.zeros(ocf.bin_centers.shape)

        # points are within distances closer than dr, so their impact on
        # the result should be minimal.
        points = [[dr / 4, 0, 0], [-dr / 4, 0, 0], [0, dr / 4, 0], [0, -dr / 4, 0]]

        test_set = util.make_raw_query_nlist_test_set(
            box, points, query_points, "ball", r_max, 0, False
        )
        for nq, neighbors in test_set:
            ocf = freud.density.CorrelationFunction(bins, r_max)
            # try for different scalar values.
            for rv in [0, 1, 2, 7]:
                values = [rv] * 4

                ocf.compute(nq, values, query_points, query_values, neighbors=neighbors)

                npt.assert_allclose(ocf.correlation, expected_correlation, atol=1e-6)

    @pytest.fixture(scope="session")
    def correlation_calc(self):
        def value_func(_r):
            return np.sin(_r)

        r_max = 10.0
        bins = 100

        ocf = freud.density.CorrelationFunction(bins, r_max)

        query_points = []
        query_values = []
        expected_correlation = []
        N = 300

        # We are generating the values so that they are sine wave from 0 to 2pi
        # rotated around z axis.  Therefore, the correlation should be a scalar
        # multiple sin if we set our ref_point to be in the origin.
        for r in ocf.bin_centers:
            for k in range(N):
                query_points.append(
                    [r * np.cos(2 * np.pi * k / N), r * np.sin(2 * np.pi * k / N), 0]
                )
                query_values.append(value_func(r))
            expected_correlation.append(value_func(r))

        expected_correlation = np.array(expected_correlation)
        return (query_points, query_values, expected_correlation)

    @pytest.mark.parametrize("rv", [0, 1, 2, 7])
    def test_points_ne_query_points_real(self, rv, correlation_calc):
        r_max = 10.0
        bins = 100
        dr = r_max / bins
        box_size = r_max * 5
        box = freud.box.Box.square(box_size)

        ocf = freud.density.CorrelationFunction(bins, r_max)

        query_points, query_values, expected_correlation = correlation_calc

        # points are within distances closer than dr, so their impact on
        # the result should be minimal.
        points = [[dr / 4, 0, 0], [-dr / 4, 0, 0], [0, dr / 4, 0], [0, -dr / 4, 0]]

        test_set = util.make_raw_query_nlist_test_set(
            box, points, query_points, "ball", r_max, 0, False
        )
        for nq, neighbors in test_set:
            ocf = freud.density.CorrelationFunction(bins, r_max)
            # try for different scalar values.
            values = [rv] * 4

            ocf.compute(nq, values, query_points, query_values, neighbors=neighbors)
            expected = expected_correlation * rv

            npt.assert_allclose(ocf.correlation, expected, atol=1e-6)


class TestCorrelationFunctionManagedArray(ManagedArrayTestBase):
    def build_object(self):
        self.obj = freud.density.CorrelationFunction(50, 3)

    @property
    def computed_properties(self):
        return ["bin_counts", "correlation"]

    def compute(self):
        box = freud.box.Box.cube(10)
        num_points = 100
        points = np.random.rand(num_points, 3) * box.L - box.L / 2
        values = np.random.rand(num_points) + np.random.rand(num_points) * 1j
        self.obj.compute((box, points), values, neighbors={"r_max": 3})
