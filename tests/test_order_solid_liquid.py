# Copyright (c) 2010-2025 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import conftest
import matplotlib
import numpy.testing as npt
import pytest

import freud

matplotlib.use("agg")


class TestSolidLiquid:
    def test_shape(self):
        N = 1000
        L = 10

        box, positions = freud.data.make_random_system(L, N, seed=1)

        comp = freud.order.SolidLiquid(6, q_threshold=0.7, solid_threshold=6)
        comp.compute((box, positions), neighbors=dict(r_max=2.0))

        npt.assert_equal(comp.cluster_idx.shape, (N,))

    def test_nlist(self):
        """Check that the internally generated NeighborList is correct."""
        N = 1000
        L = 10

        box, positions = freud.data.make_random_system(L, N, seed=1)

        query_args = dict(r_max=2.0, exclude_ii=True)
        comp = freud.order.SolidLiquid(6, q_threshold=0.7, solid_threshold=6).compute(
            (box, positions), neighbors=query_args
        )

        aq = freud.locality.AABBQuery(box, positions)
        nlist = aq.query(positions, query_args).toNeighborList()

        npt.assert_array_equal(nlist[:], comp.nlist[:])

    @pytest.mark.parametrize(
        ("comp", "query_args"),
        [
            (comp, query_args)
            for comp in (
                freud.order.SolidLiquid(6, q_threshold=0.7, solid_threshold=6),
                freud.order.SolidLiquid(
                    6, q_threshold=0.3, solid_threshold=6, normalize_q=False
                ),
            )
            for query_args in (dict(r_max=2.0), dict(num_neighbors=12))
        ],
    )
    def test_identical_environments(self, comp, query_args):
        box, positions = freud.data.UnitCell.fcc().generate_system(4, scale=2)

        comp.compute((box, positions), neighbors=query_args)
        assert comp.largest_cluster_size == len(positions)
        assert len(comp.cluster_sizes) == 1
        assert comp.cluster_sizes[0] == len(positions)
        npt.assert_array_equal(comp.num_connections, 12)

    def test_multiple_compute(self):
        # Covers the case where compute is called multiple times
        box, positions = freud.data.UnitCell.fcc().generate_system(4, scale=2)

        comp_default = freud.order.SolidLiquid(6, q_threshold=0.7, solid_threshold=6)
        for query_args in (dict(r_max=2.0), dict(num_neighbors=12)):
            comp_default.compute((box, positions), neighbors=query_args)
            assert comp_default.largest_cluster_size == len(positions)
            assert len(comp_default.cluster_sizes) == 1
            assert comp_default.cluster_sizes[0] == len(positions)
            npt.assert_array_equal(comp_default.num_connections, 12)

    def test_nlist_lifetime(self):
        def _get_nlist(sys):
            sl = freud.order.SolidLiquid(2, 0.5, 0.2)
            sl.compute(sys, neighbors=dict(r_max=2))
            return sl.nlist

        conftest.nlist_lifetime_check(_get_nlist)

    def test_attribute_access(self):
        box, positions = freud.data.UnitCell.fcc().generate_system(4, scale=2)
        sph_l = 6
        q_threshold = 0.7
        solid_threshold = 4
        normalize_q = False

        comp = freud.order.SolidLiquid(
            sph_l,
            q_threshold=q_threshold,
            solid_threshold=solid_threshold,
            normalize_q=normalize_q,
        )

        assert comp.l == sph_l
        npt.assert_allclose(comp.q_threshold, q_threshold)
        npt.assert_allclose(comp.solid_threshold, solid_threshold)
        assert comp.normalize_q == normalize_q

        with pytest.raises(AttributeError):
            comp.largest_cluster_size
        with pytest.raises(AttributeError):
            comp.cluster_sizes
        with pytest.raises(AttributeError):
            comp.cluster_idx
        with pytest.raises(AttributeError):
            comp.num_connections
        with pytest.raises(AttributeError):
            comp.ql_ij
        with pytest.raises(AttributeError):
            comp.plot()

        comp.compute((box, positions), neighbors=dict(r_max=2.0))

        comp.largest_cluster_size
        comp.cluster_sizes
        comp.cluster_idx
        comp.num_connections
        comp.ql_ij
        comp._repr_png_()

    def test_repr(self):
        comp = freud.order.SolidLiquid(6, q_threshold=0.7, solid_threshold=6)
        assert str(comp) == str(eval(repr(comp)))
