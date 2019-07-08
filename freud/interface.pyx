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

    .. moduleauthor:: Matthew Spellings <mspells@umich.edu>
    .. moduleauthor:: Bradley Dice <bdice@bradleydice.com>

    Args:
        box (:class:`freud.box.Box`): Simulation box.
        r_cut (float): Distance to search for particle neighbors.

    Attributes:
        ref_point_count (int):
            Number of particles from :code:`ref_points` on the interface.
        ref_point_ids (:class:`np.ndarray`):
            The particle IDs from :code:`ref_points`.
        point_count (int):
            Number of particles from :code:`points` on the interface.
        point_ids (:class:`np.ndarray`):
            The particle IDs from :code:`points`.
    """
    cdef float rmax
    cdef const unsigned int[::1] _ref_point_ids
    cdef const unsigned int[::1] _point_ids

    def __cinit__(self, float r_cut):
        self.rmax = r_cut
        self._ref_point_ids = np.empty(0, dtype=np.uint32)
        self._point_ids = np.empty(0, dtype=np.uint32)

    @Compute._compute()
    def compute(self, box, ref_points, points, nlist=None):
        R"""Compute the particles at the interface between the two given sets of
        points.

        Args:
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                One set of particle positions.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Other set of particle positions.
            nlist (:class:`freud.locality.NeighborList`, optional):
                Neighborlist to use to find bonds (Default value = None).
        """
        b = freud.common.convert_box(box)
        ref_points = freud.common.convert_array(ref_points, shape=(None, 3))
        points = freud.common.convert_array(points, shape=(None, 3))

        if nlist is None:
            lc = freud.locality.LinkCell(b, self.rmax)
            nlist = lc.compute(b, ref_points, points).nlist
        else:
            nlist = nlist.copy().filter_r(b, ref_points, points, self.rmax)

        self._ref_point_ids = np.unique(nlist.index_i).astype(np.uint32)
        self._point_ids = np.unique(nlist.index_j).astype(np.uint32)
        return self

    @Compute._computed_property()
    def ref_point_count(self):
        return len(self._ref_point_ids)

    @Compute._computed_property()
    def ref_point_ids(self):
        return np.asarray(self._ref_point_ids)

    @Compute._computed_property()
    def point_count(self):
        return len(self._point_ids)

    @Compute._computed_property()
    def point_ids(self):
        return np.asarray(self._point_ids)

    def __repr__(self):
        return "freud.interface.{cls}(r_cut={r_cut})".format(
            cls=type(self).__name__, r_cut=self.rmax)

    def __str__(self):
        return repr(self)
