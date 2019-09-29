# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

R"""
The :class:`freud.interface` module contains functions to measure the interface
between sets of points.
"""

import freud.common
import numpy as np

from freud.common cimport Compute
from freud.util cimport vec3
from cython.operator cimport dereference
import freud.locality

cimport freud.locality
cimport freud.box

cimport numpy as np

# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef class InterfaceMeasure(Compute):
    R"""Measures the interface between two sets of points.

    Args:
        box (:class:`freud.box.Box`): Simulation box.
        r_max (float): Distance to search for particle neighbors.

    Attributes:
        point_count (int):
            Number of particles from :code:`points` on the interface.
        point_ids (:class:`np.ndarray`):
            The particle IDs from :code:`points`.
        query_point_count (int):
            Number of particles from :code:`query_points` on the interface.
        query_point_ids (:class:`np.ndarray`):
            The particle IDs from :code:`query_points`.
    """
    cdef float r_max
    cdef const unsigned int[::1] _point_ids
    cdef const unsigned int[::1] _query_point_ids

    def __cinit__(self, float r_max):
        self.r_max = r_max
        self._point_ids = np.empty(0, dtype=np.uint32)
        self._query_point_ids = np.empty(0, dtype=np.uint32)

    @Compute._compute()
    def compute(self, box, points, query_points, nlist=None):
        R"""Compute the particles at the interface between the two given sets of
        points.

        Args:
            points ((:math:`N_{points}`, 3) :class:`numpy.ndarray`):
                One set of particle positions.
            query_points ((:math:`N_{query_points}`, 3) :class:`numpy.ndarray`):
                Other set of particle positions.
            nlist (:class:`freud.locality.NeighborList`, optional):
                Neighborlist to use to find bonds (Default value = None).
        """  # noqa E501
        b = freud.common.convert_box(box)
        points = freud.common.convert_array(points, shape=(None, 3))
        query_points = freud.common.convert_array(
            query_points, shape=(None, 3))

        if nlist is None:
            aq = freud.locality.AABBQuery(b, points)
            nlist = aq.query(query_points,
                             dict(r_max=self.r_max)).toNeighborList()
        else:
            nlist = nlist.copy().filter_r(self.r_max)

        self._point_ids = np.unique(nlist.point_indices)
        self._query_point_ids = np.unique(nlist.query_point_indices)
        return self

    @Compute._computed_property()
    def point_count(self):
        return len(self._point_ids)

    @Compute._computed_property()
    def point_ids(self):
        return np.asarray(self._point_ids)

    @Compute._computed_property()
    def query_point_count(self):
        return len(self._query_point_ids)

    @Compute._computed_property()
    def query_point_ids(self):
        return np.asarray(self._query_point_ids)

    def __repr__(self):
        return "freud.interface.{cls}(r_max={r_max})".format(
            cls=type(self).__name__, r_max=self.r_max)
