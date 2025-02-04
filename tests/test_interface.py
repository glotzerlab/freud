# Copyright (c) 2010-2025 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import numpy as np
import pytest

import freud


class TestInterface:
    def test_constructor(self):
        """Ensure no arguments are accepted to the constructor."""
        freud.interface.Interface()
        with pytest.raises(TypeError):
            freud.interface.Interface(0)

    def test_take_one(self):
        """Test that there is exactly 1 or 12 particles at the interface when
        one particle is removed from an FCC structure"""
        np.random.seed(0)
        (box, positions) = freud.data.UnitCell.fcc().generate_system(
            4, scale=2, sigma_noise=1e-2
        )
        positions.flags["WRITEABLE"] = False

        index = np.random.randint(0, len(positions))

        point = positions[index].reshape((1, 3))
        others = np.concatenate([positions[:index], positions[index + 1 :]])

        inter = freud.interface.Interface()

        # Test attribute access
        with pytest.raises(AttributeError):
            inter.point_count
        with pytest.raises(AttributeError):
            inter.point_ids
        with pytest.raises(AttributeError):
            inter.query_point_count
        with pytest.raises(AttributeError):
            inter.query_point_ids

        test_one = inter.compute((box, point), others, neighbors=dict(r_max=1.5))

        # Test attribute access
        inter.point_count
        inter.point_ids
        inter.query_point_count
        inter.query_point_ids

        assert test_one.point_count == 1
        assert len(test_one.point_ids) == 1

        test_twelve = inter.compute((box, others), point, neighbors=dict(r_max=1.5))
        assert test_twelve.point_count == 12
        assert len(test_twelve.point_ids) == 12

    def test_filter_r(self):
        """Test that nlists are filtered to the correct r_max."""
        np.random.seed(0)
        r_max = 3.0
        (box, positions) = freud.data.UnitCell.fcc().generate_system(
            4, scale=2, sigma_noise=1e-2
        )

        index = np.random.randint(0, len(positions))

        point = positions[index].reshape((1, 3))
        others = np.concatenate([positions[:index], positions[index + 1 :]])

        # Creates a NeighborList with r_max larger than the interface size
        aq = freud.locality.AABBQuery(box, others)
        nlist = aq.query(point, dict(r_max=r_max)).toNeighborList()

        # Filter NeighborList
        nlist.filter_r(1.5)

        inter = freud.interface.Interface()

        test_twelve = inter.compute((box, others), point, nlist)
        assert test_twelve.point_count == 12
        assert len(test_twelve.point_ids) == 12

    def test_repr(self):
        inter = freud.interface.Interface()
        assert str(inter) == str(eval(repr(inter)))
