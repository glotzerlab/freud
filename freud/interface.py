# Copyright (c) 2010-2024 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

r"""
The :class:`freud.interface` module contains functions to measure the interface
between sets of points.
"""

from freud.locality import _PairCompute, _make_default_nlist
from freud.util import _Compute

import numpy as np


class Interface(_PairCompute):
    r"""Measures the interface between two sets of points."""

    def __init__(self):
        self._point_ids = np.empty(0, dtype=np.uint32)
        self._query_point_ids = np.empty(0, dtype=np.uint32)

    def compute(self, system, query_points, neighbors=None):
        r"""Compute the particles at the interface between two sets of points.

        Args:
            system:
                Any object that is a valid argument to
                :class:`freud.locality.NeighborQuery.from_system`.
            query_points ((:math:`N_{query\_points}`, 3) :class:`numpy.ndarray`, optional):
                Second set of points (in addition to the system points) to
                calculate the interface.
            neighbors (:class:`freud.locality.NeighborList` or dict, optional):
                Either a :class:`NeighborList <freud.locality.NeighborList>` of
                neighbor pairs to use in the calculation, or a dictionary of
                `query arguments
                <https://freud.readthedocs.io/en/stable/topics/querying.html>`_
                (Default value: None).
        """  # noqa E501

        # freud.locality.
        nlist = _make_default_nlist(system, neighbors, query_points)

        self._point_ids = np.unique(nlist.point_indices)
        self._query_point_ids = np.unique(nlist.query_point_indices)
        return self

    @_Compute._computed_property
    def point_count(self):
        """int: Number of particles from :code:`points` on the interface."""
        return len(self._point_ids)

    @_Compute._computed_property
    def point_ids(self):
        """:class:`np.ndarray`: The particle IDs from :code:`points`."""
        return np.asarray(self._point_ids)

    @_Compute._computed_property
    def query_point_count(self):
        """int: Number of particles from :code:`query_points` on the
        interface."""
        return len(self._query_point_ids)

    @_Compute._computed_property
    def query_point_ids(self):
        """:class:`np.ndarray`: The particle IDs from :code:`query_points`."""
        return np.asarray(self._query_point_ids)

    def __repr__(self):
        return f"freud.interface.{type(self).__name__}()"