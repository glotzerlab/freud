# Copyright (c) 2010-2026 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import numpy as np
import pytest

import freud

spatula = pytest.importorskip("spatula")


class TestPGOP:
    def test_defaults(self):
        pgop = freud.environment.PGOP(["Oh", "D3h"])
        assert pgop.symmetries == ["Oh", "D3h"]
        assert pgop.mesh_quaternions == 600
        assert (
            repr(pgop)
            == "freud.environment.PGOP(symmetries=['Oh', 'D3h'], mesh_quaternions=600)"
        )

    def test_compute_perfect_crystals(self):
        """The correct point group symmetry scores highest for each crystal."""
        systems = {
            "fcc": (freud.data.UnitCell.fcc().generate_system(4, seed=0), 1.5),
            "hcp": (freud.data.UnitCell.hcp().generate_system(4), 1.4),
        }
        pgop = freud.environment.PGOP(["Oh", "D3h"])
        for system, (crystal, r_max) in systems.items():
            pgop.compute(crystal, sigma=0.075, neighbors={"r_max": r_max})
            order = pgop.order
            assert order.shape == (2,)
            assert (order > 0).all()
            assert (order <= 1.0001).all()
            # The site symmetry of fcc is Oh while that of hcp is D3h. Both
            # crystals are aligned with the box, so their optimal orientation
            # is the identity rotation, which is part of the default mesh.
            symmetry_index = {"fcc": 0, "hcp": 1}[system]
            np.testing.assert_allclose(order[symmetry_index], 1.0, atol=1e-5)
            assert order[symmetry_index] == np.max(order)

    def test_order_is_average_of_particle_order(self):
        fcc = freud.data.UnitCell.fcc().generate_system(4, seed=0)
        pgop = freud.environment.PGOP(["Oh", "D3h"], mesh_quaternions=100)
        assert pgop.compute(fcc, 0.075, {"r_max": 1.5, "exclude_ii": True}) is pgop
        assert pgop.particle_order.shape == (len(fcc[1]), 2)
        np.testing.assert_allclose(
            pgop.order, np.mean(pgop.particle_order, axis=0), atol=1e-6
        )

    def test_uncomputed_properties_error(self):
        pgop = freud.environment.PGOP(["Oh"])
        with pytest.raises(AttributeError):
            pgop.order
        with pytest.raises(AttributeError):
            pgop.particle_order

    def test_no_default_query_args(self):
        fcc = freud.data.UnitCell.fcc().generate_system(4, seed=0)
        pgop = freud.environment.PGOP(["Oh"], mesh_quaternions=50)
        with pytest.raises(NotImplementedError):
            pgop.compute(fcc, 0.075)
        with pytest.raises(ValueError):
            pgop.compute(fcc, 0.075, neighbors=5)

    def test_query_points(self):
        fcc = freud.data.UnitCell.fcc().generate_system(4, seed=0)
        _box, points = fcc
        query_points = np.asarray(points)[::7]
        pgop = freud.environment.PGOP(["Oh"], mesh_quaternions=50)
        pgop.compute(fcc, 0.075, {"r_max": 1.5}, query_points=query_points)
        assert pgop.particle_order.shape == (len(query_points), 1)

    def test_set_mesh_quaternions(self):
        """The mesh quaternions may be an int or an array of quaternions."""
        pgop = freud.environment.PGOP(["Oh"], mesh_quaternions=100)
        assert pgop.mesh_quaternions == 100
        pgop.mesh_quaternions = 200
        assert pgop.mesh_quaternions == 200
        with pytest.raises(ValueError, match="must be an int"):
            pgop.mesh_quaternions = None

        # A (N, 4) array of quaternions is accepted and normalized.
        orientations = [[2, 0, 0, 0], [0, 3, 0, 0]]
        pgop.mesh_quaternions = orientations
        assert pgop.mesh_quaternions.shape == (2, 4)
        np.testing.assert_allclose(
            pgop.mesh_quaternions, [[1, 0, 0, 0], [0, 1, 0, 0]], atol=1e-12
        )
        fcc = freud.data.UnitCell.fcc().generate_system(3, seed=0)
        pgop.compute(fcc, 0.075, {"r_max": 1.3, "exclude_ii": True})
        assert (pgop.particle_order > 0).all()

    def test_explicit_mesh_quaternions(self):
        """An explicit mesh is searched exactly as provided."""
        fcc = freud.data.UnitCell.fcc().generate_system(3, seed=0)
        neighbors = {"r_max": 1.3, "exclude_ii": True}

        # A (4,) quaternion is expanded to (1, 4) and normalized.
        pgop = freud.environment.PGOP(["Oh"], mesh_quaternions=[2, 0, 0, 0])
        np.testing.assert_allclose(
            pgop.mesh_quaternions, [[1.0, 0.0, 0.0, 0.0]], atol=1e-12
        )
        # The identity rotation is optimal for a box-aligned perfect crystal.
        pgop.compute(fcc, 0.075, neighbors)
        np.testing.assert_allclose(pgop.order[0], 1.0, atol=1e-5)

        # A mesh of only the identity quaternion is equivalent to an int mesh
        # of 1 (the identity rotation followed by an empty lattice).
        identity_only = pgop.order.copy()
        pgop.mesh_quaternions = 1
        pgop.compute(fcc, 0.075, neighbors)
        np.testing.assert_allclose(pgop.order, identity_only, atol=1e-6)

    def test_invalid_arguments(self):
        with pytest.raises(TypeError, match="sequence of str"):
            freud.environment.PGOP("Oh")
        with pytest.raises(ValueError, match="must be positive"):
            freud.environment.PGOP(["Oh"], mesh_quaternions=0)
        with pytest.raises(ValueError, match="must be positive"):
            freud.environment.PGOP(["Oh"], mesh_quaternions=-1)
        with pytest.raises(ValueError, match="must be an int"):
            freud.environment.PGOP(["Oh"], mesh_quaternions="many")
        with pytest.raises(ValueError, match="must be an int"):
            freud.environment.PGOP(["Oh"], mesh_quaternions=None)
        with pytest.raises(ValueError, match=r"shape \(4,\)"):
            freud.environment.PGOP(["Oh"], mesh_quaternions=[1, 0, 0])
        with pytest.raises(ValueError, match="nonzero quaternion"):
            freud.environment.PGOP(["Oh"], mesh_quaternions=[0, 0, 0, 0])
        with pytest.raises(ValueError, match="nonzero quaternion"):
            freud.environment.PGOP(["Oh"], mesh_quaternions=np.empty((0, 4)))

    def test_nan_for_particles_without_neighbors(self):
        box, points = freud.data.make_random_system(10, 50, seed=5)
        pgop = freud.environment.PGOP(["Oh"], mesh_quaternions=50)
        pgop.compute((box, points), 0.075, {"r_max": 1.0, "exclude_ii": True})
        isolated = np.isnan(pgop.particle_order[:, 0])
        assert isolated.any()
        assert isolated.sum() > 10
