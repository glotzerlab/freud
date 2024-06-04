# Copyright (c) 2010-2024 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pytest
import util
from numpy.lib import NumpyVersion
from test_managedarray import ManagedArrayTestBase

import freud

matplotlib.use("agg")


class TestRDF:
    @pytest.mark.parametrize("r_min", [0, 0.05, 0.1, 1.0, 3.0])
    def test_generate_bins(self, r_min):
        r_max = 5
        bins = round((r_max - r_min) / 0.1)
        dr = (r_max - r_min) / bins

        # make sure the radius for each bin is generated correctly
        r_list = np.array(
            [
                r_min + dr * (i + 1 / 2)
                for i in range(bins)
                if r_min + dr * (i + 1 / 2) < r_max
            ]
        )
        rdf = freud.density.RDF(bins, r_max, r_min=r_min)
        npt.assert_allclose(rdf.bin_centers, r_list, rtol=1e-4, atol=1e-4)
        npt.assert_allclose((rdf.bin_edges + dr / 2)[:-1], r_list, rtol=1e-4, atol=1e-4)

    def test_attribute_access(self):
        r_max = 10.0
        bins = 10
        num_points = 100
        box_size = r_max * 3.1
        box, points = freud.data.make_random_system(box_size, num_points, is2D=True)
        rdf = freud.density.RDF(r_max=r_max, bins=bins)

        # Test protected attribute access
        with pytest.raises(AttributeError):
            rdf.rdf
        with pytest.raises(AttributeError):
            rdf.box
        with pytest.raises(AttributeError):
            rdf.n_r

        rdf.compute((box, points), reset=False)

        # Test if accessible now
        rdf.rdf
        rdf.box
        rdf.n_r

        rdf.compute((box, points))

        # Test if accessible now
        rdf.rdf
        rdf.box
        rdf.n_r

    def test_invalid_rdf(self):
        # Make sure that invalid RDF objects raise errors
        with pytest.raises(ValueError):
            freud.density.RDF(r_max=-1, bins=10)
        with pytest.raises(ValueError):
            freud.density.RDF(r_max=1, bins=0)
        with pytest.raises(ValueError):
            freud.density.RDF(r_max=1, bins=10, r_min=2)
        with pytest.raises(ValueError):
            freud.density.RDF(r_max=1, bins=10, r_min=-1)
        with pytest.raises(ValueError):
            freud.density.RDF(r_max=1, bins=10, r_min=-1, normalization_mode="blah")

    @pytest.mark.parametrize("mode", ["exact", "finite_size"])
    def test_normalization_mode(self, mode):
        """Make sure RDF can be computed with different normalization modes."""
        r_max = 10.0
        bins = 10
        num_points = 100
        box_size = r_max * 3.1
        sys = freud.data.make_random_system(box_size, num_points, is2D=True)
        rdf = freud.density.RDF(r_max=r_max, bins=bins, normalization_mode=mode)
        rdf.compute(sys)

    @pytest.mark.parametrize("r_min", [0, 0.1, 3.0])
    def test_random_point(self, r_min):
        r_max = 10.0
        bins = 10
        num_points = 10000
        tolerance = 0.1
        box_size = r_max * 3.1

        box, points = freud.data.make_random_system(box_size, num_points)
        # This test is slow, and since it's a validation of the underlying
        # algorithm and not the API we don't need to test all possible
        # inputs, so we only test the fastest one (AABBQuery).
        nq = freud.locality.AABBQuery(box, points)
        neighbors = {"mode": "ball", "r_max": r_max, "exclude_ii": True}

        rdf = freud.density.RDF(bins, r_max, r_min)

        rdf.compute(nq, neighbors=neighbors, reset=False)

        assert rdf.box == box
        expected = np.ones(bins, dtype=np.float32)
        npt.assert_allclose(rdf.rdf, expected, atol=tolerance)

        # Numerical integration to compute the running coordination
        # number will be highly inaccurate, so we can only test up to
        # a limited precision. Also, since dealing with nonzero r_min
        # values requires extrapolation, we only test when r_min=0.
        ndens = points.shape[0] / box.volume
        dr = (r_max - r_min) / bins
        bin_boundaries = np.array(
            [r_min + dr * i for i in range(bins + 1) if r_min + dr * i <= r_max]
        )
        bin_volumes = 4 / 3 * np.pi * np.diff(bin_boundaries**3)
        avg_counts = rdf.rdf * ndens * bin_volumes
        npt.assert_allclose(rdf.n_r, np.cumsum(avg_counts), rtol=tolerance)

    def test_compute_reset(self):
        # This test is to check whether rdf.compute accumulates the data correctly
        # when reset is set to False
        r_max = 10.0
        bins = 10
        num_points = 10000
        tolerance = 0.1
        box_size = r_max * 3.1

        for r_min in (0, 0.1):
            box, points = freud.data.make_random_system(box_size, num_points)
            # This test is slow, and since it's a validation of the underlying
            # algorithm and not the API we don't need to test all possible
            # inputs, so we only test the fastest one (AABBQuery).
            nq = freud.locality.AABBQuery(box, points)
            neighbors = {"mode": "ball", "r_max": r_max, "exclude_ii": True}

            rdf = freud.density.RDF(bins, r_max, r_min)

            rdf.compute(nq, neighbors=neighbors, reset=False)

            assert rdf.box == box
            expected = np.ones(bins, dtype=np.float32)
            npt.assert_allclose(rdf.rdf, expected, atol=tolerance)

            # Numerical integration to compute the running coordination
            # number will be highly inaccurate, so we can only test up to
            # a limited precision. Also, since dealing with nonzero r_min
            # values requires extrapolation, we only test when r_min=0.
            ndens = points.shape[0] / box.volume
            dr = (r_max - r_min) / bins
            bin_boundaries = np.array(
                [r_min + dr * i for i in range(bins + 1) if r_min + dr * i <= r_max]
            )
            bin_volumes = 4 / 3 * np.pi * np.diff(bin_boundaries**3)
            avg_counts = rdf.rdf * ndens * bin_volumes
            npt.assert_allclose(rdf.n_r, np.cumsum(avg_counts), rtol=tolerance)

    def test_repr(self):
        rdf = freud.density.RDF(r_max=10, bins=100, r_min=0.5)
        assert str(rdf) == str(eval(repr(rdf)))

    def test_repr_png(self):
        r_max = 10.0
        bins = 10
        num_points = 10
        box_size = r_max * 3.1
        box, points = freud.data.make_random_system(box_size, num_points)
        rdf = freud.density.RDF(bins, r_max)

        with pytest.raises(AttributeError):
            rdf.plot()
        assert rdf._repr_png_() is None

        rdf.compute((box, points), reset=False)
        rdf.plot()
        rdf._repr_png_()
        plt.close("all")

    def test_points_ne_query_points(self):
        r_max = 100.0
        bins = 100
        box_size = r_max * 5
        box = freud.box.Box.square(box_size)

        rdf = freud.density.RDF(bins, r_max)

        query_points = []
        supposed_n_r = [0]
        N = 100

        # With points closely centered around the origin,
        # the cumulative average bin counts should be same as
        # having a single point at the origin.
        # Also, we can check for whether points are not considered against
        # each other.
        dr = r_max / bins
        points = [[dr / 4, 0, 0], [-dr / 4, 0, 0], [0, dr / 4, 0], [0, -dr / 4, 0]]
        num_points = len(points)
        for r in rdf.bin_centers:
            for k in range(N):
                query_points.append(
                    [r * np.cos(2 * np.pi * k / N), r * np.sin(2 * np.pi * k / N), 0]
                )
            supposed_n_r.append(supposed_n_r[-1] + num_points * N)
        supposed_n_r = np.array(supposed_n_r[1:])
        supposed_n_r = supposed_n_r / len(query_points)

        test_set = util.make_raw_query_nlist_test_set(
            box, points, query_points, "ball", r_max, 0, False
        )
        for nq, neighbors in test_set:
            rdf = freud.density.RDF(bins, r_max)
            rdf.compute(nq, query_points, neighbors=neighbors)

            npt.assert_allclose(rdf.n_r, supposed_n_r, atol=1e-5, rtol=1e-6)

    def test_empty_histogram(self):
        r_max = 0.5
        bins = 10
        box_size = 5
        box = freud.box.Box.cube(box_size)
        rdf = freud.density.RDF(bins, r_max)
        points = [[0, 0, 0], [2, 2, 2]]
        rdf.compute(system=(box, points))

        # Test that properties are accessible even though there's no data
        npt.assert_array_equal(rdf.rdf, np.zeros(bins))
        npt.assert_array_equal(rdf.n_r, np.zeros(bins))

    @pytest.mark.skipif(
        NumpyVersion(np.__version__) < "1.15.0", reason="Requires numpy>=1.15.0."
    )
    def test_bin_precision(self):
        # Ensure bin edges are precise
        bins = 500
        r_min = 0
        r_max = 50
        rdf = freud.density.RDF(bins=bins, r_max=r_max, r_min=r_min)
        expected_bin_edges = np.histogram_bin_edges(
            np.array([0], dtype=np.float32), bins=bins, range=[r_min, r_max]
        )
        npt.assert_allclose(rdf.bin_edges, expected_bin_edges, atol=1e-6)


class TestRDFManagedArray(ManagedArrayTestBase):
    def build_object(self):
        self.obj = freud.density.RDF(50, 3)

    @property
    def computed_properties(self):
        return ["rdf", "n_r", "bin_counts"]

    def compute(self):
        box = freud.box.Box.cube(10)
        num_points = 100
        points = np.random.rand(num_points, 3) * box.L - box.L / 2
        self.obj.compute((box, points), neighbors={"r_max": 2})
