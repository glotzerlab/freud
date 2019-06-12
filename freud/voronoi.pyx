# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

R"""
The :class:`freud.voronoi` module contains tools to characterize Voronoi cells
of a system.
"""

import numpy as np
import logging
import copy
import freud.common
import warnings

from freud.common cimport Compute
from freud.util._VectorMath cimport vec3
from cython.operator cimport dereference

cimport freud.box
cimport freud.locality
cimport numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy.spatial import Voronoi as qvoronoi
    from scipy.spatial import ConvexHull
    from scipy.sparse import csr_matrix
    _SCIPY_AVAILABLE = True
except ImportError:
    qvoronoi = None
    msg = ('scipy.spatial.Voronoi is not available (requires scipy 0.12+), '
           'so freud.voronoi is not available.')
    logger.warning(msg)
    _SCIPY_AVAILABLE = False


# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()


class Voronoi(Compute):
    R"""Compute the Voronoi tessellation of a 2D or 3D system using qhull.
    This uses :class:`scipy.spatial.Voronoi`, accounting for periodic
    boundary conditions.

    .. moduleauthor:: Benjamin Schultz <baschult@umich.edu>
    .. moduleauthor:: Yina Geng <yinageng@umich.edu>
    .. moduleauthor:: Mayank Agrawal <amayank@umich.edu>
    .. moduleauthor:: Bradley Dice <bdice@bradleydice.com>

    Since qhull does not support periodic boundary conditions natively, we
    expand the box to include a portion of the particles' periodic images.
    The buffer width is given by the parameter :code:`buff`. The
    computation of Voronoi tessellations and neighbors is only guaranteed
    to be correct if :code:`buff >= L/2` where :code:`L` is the longest side
    of the simulation box. For dense systems with particles filling the
    entire simulation volume, a smaller value for :code:`buff` is acceptable.
    If the buffer width is too small, then some polytopes may not be closed
    (they may have a boundary at infinity), and these polytopes' vertices are
    excluded from the list.  If either the polytopes or volumes lists that are
    computed is different from the size of the array of positions used in the
    :meth:`freud.voronoi.Voronoi.compute()` method, try recomputing using a
    larger buffer width.

    Args:
        box (:class:`freud.box.Box`):
            Simulation box.
        buff (float):
            Buffer width.

    Attributes:
        buffer (float):
            Buffer width.
        nlist (:class:`~.locality.NeighborList`):
            Returns a weighted neighbor list.  In 2D systems, the bond weight
            is the "ridge length" of the Voronoi boundary line between the
            neighboring particles.  In 3D systems, the bond weight is the
            "ridge area" of the Voronoi boundary polygon between the
            neighboring particles.
        polytopes (list[:class:`numpy.ndarray`]):
            List of arrays, each containing Voronoi polytope vertices.
        volumes ((:math:`\left(N_{cells} \right)`) :class:`numpy.ndarray`):
            Returns an array of volumes (areas in 2D) corresponding to Voronoi
            cells.
    """

    def __init__(self, box, buff=0.1):
        if not _SCIPY_AVAILABLE:
            raise RuntimeError("You cannot use this class without SciPy")
        cdef freud.box.Box b = freud.common.convert_box(box)
        self._box = b
        self._buff = buff
        self._nlist = freud.locality.NeighborList()
        self._set_compute_flag("compute")
        self._set_compute_flag("computeVolumes")
        self._set_compute_flag("computeNeighbors")

    def _qhull_compute(self, positions, box=None, buff=None):
        R"""Calls ParticleBuffer and qhull

        Args:
            positions ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate Voronoi diagram for.
            box (:class:`freud.box.Box`):
                Simulation box (Default value = None).
            buff (float):
                Buffer distance within which to look for images
                (Default value = None).
        """
        # Compute the buffer particles in C++
        pbuff = freud.box.ParticleBuffer(box)
        pbuff.compute(positions, buff)
        buff_ptls = pbuff.buffer_particles
        buff_ids = pbuff.buffer_ids

        if buff_ptls.size > 0:
            self.expanded_points = np.concatenate((positions, buff_ptls))
            self.expanded_ids = np.concatenate((
                np.arange(len(positions)), buff_ids))
        else:
            self.expanded_points = positions
            self.expanded_ids = np.arange(len(positions))

        # Use only the first two components if the box is 2D
        if box.is2D():
            self.expanded_points = self.expanded_points[:, :2]

        # Use qhull to get the points
        self.voronoi = qvoronoi(self.expanded_points)

    @Compute._compute("compute")
    def compute(self, positions, box=None, buff=None):
        R"""Compute Voronoi diagram.

        Args:
            positions ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate Voronoi diagram for.
            box (:class:`freud.box.Box`):
                Simulation box (Default value = None).
            buff (float):
                Buffer distance within which to look for images
                (Default value = None).
        """

        # If box or buff is not specified, revert to object quantities
        cdef freud.box.Box b
        if box is None:
            b = self._box
        else:
            b = freud.common.convert_box(box)
        if buff is None:
            buff = self._buff

        self._qhull_compute(positions, b, buff)

        vertices = self.voronoi.vertices

        # Add a z-component of 0 if the box is 2D
        if b.is2D():
            vertices = np.insert(vertices, 2, 0, 1)

        # Construct a list of polytope vertices
        self._poly_verts = list()
        for region in self.voronoi.point_region[:len(positions)]:
            if -1 in self.voronoi.regions[region]:
                continue
            self._poly_verts.append(vertices[self.voronoi.regions[region]])
        return self

    @property
    def buffer(self):
        return self._buff

    @Compute._computed_property("compute")
    def polytopes(self):
        R"""Returns a list of polytope vertices corresponding to Voronoi cells.

        If the buffer width is too small, then some polytopes may not be
        closed (they may have a boundary at infinity), and these polytopes'
        vertices are excluded from the list.

        The length of the list returned by this method should be the same
        as the array of positions used in the
        :meth:`freud.voronoi.Voronoi.compute()` method, if all the polytopes
        are closed. Otherwise try using a larger buffer width.

        Returns:
            list:
                List of :class:`numpy.ndarray` containing Voronoi polytope
                vertices.
        """
        return self._poly_verts

    @Compute._compute("computeNeighbors")
    def computeNeighbors(self, positions, box=None, buff=None,
                         exclude_ii=True):
        R"""Compute the neighbors of each particle based on the Voronoi
        tessellation. One can include neighbors from multiple Voronoi shells by
        specifying :code:`numShells` in :meth:`~.getNeighbors()`.
        An example of computing neighbors from the first two Voronoi shells
        for a 2D mesh is shown below.

        Retrieve the results with :meth:`~.getNeighbors()`.

        Example::

            from freud import box, voronoi
            import numpy as np
            vor = voronoi.Voronoi(box.Box(5, 5, is2D=True))
            pos = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0],
                            [1, 0, 0], [1, 1, 0], [1, 2, 0],
                            [2, 0, 0], [2, 1, 0], [2, 2, 0]], dtype=np.float32)
            first_shell = vor.computeNeighbors(pos).getNeighbors(1)
            second_shell = vor.computeNeighbors(pos).getNeighbors(2)
            print('First shell:', first_shell)
            print('Second shell:', second_shell)

        .. note:: Input positions must be a 3D array. For 2D, set the z value
                  to 0.

        Args:
            positions ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate Voronoi diagram for.
            box (:class:`freud.box.Box`):
                Simulation box (Default value = None).
            buff (float):
                Buffer distance within which to look for images
                (Default value = None).
            exclude_ii (bool, optional):
                True if pairs of points with identical indices should be
                excluded (Default value = True).
        """
        # If box or buff is not specified, revert to object quantities
        cdef freud.box.Box b
        if box is None:
            b = self._box
        else:
            b = freud.common.convert_box(box)
        if buff is None:
            buff = self._buff

        self._qhull_compute(positions, b, buff)

        cdef np.ndarray[int, ndim=2] ridge_points = self.voronoi.ridge_points
        ridge_vertices = self.voronoi.ridge_vertices

        # Must keep this in double precision
        cdef np.ndarray[np.float64_t, ndim=2] vor_vertices = \
            self.voronoi.vertices
        cdef unsigned int N = len(positions)

        cdef unsigned int index_i
        cdef unsigned int index_j
        cdef unsigned int k
        cdef np.ndarray[np.float64_t, ndim=2] vertex_coords

        # Dictionary of neighbors where keys are neighbor indices (i, j)
        # and values are bond weights (maximum ridge area shared by (i, j))
        all_bonds = {}
        for (k, (index_i, index_j)) in enumerate(ridge_points):

            if index_i >= N and index_j >= N:
                # Ignore the ridges between two buffer particles
                continue

            index_i = self.expanded_ids[index_i]
            index_j = self.expanded_ids[index_j]

            if exclude_ii and index_i == index_j:
                continue

            if -1 not in ridge_vertices[k]:
                if b.is2D():
                    # The weight for a 2D system is the
                    # length of the ridge line
                    weight = np.linalg.norm(
                        vor_vertices[ridge_vertices[k][0]] -
                        vor_vertices[ridge_vertices[k][1]])
                else:
                    # The weight for a 3D system is the ridge polygon area
                    # The process to compute this area is:
                    # 1. Project 3D polygon onto xy, yz, or zx plane,
                    #    by aligning with max component of the normal vector
                    # 2. Use shoelace formula to compute 2D area
                    # 3. Project back to get true area of 3D polygon
                    # See link below for sample code and further explanation
                    # http://geomalgorithms.com/a01-_area.html#area3D_Polygon()
                    vertex_coords = np.array(
                        [vor_vertices[i] for i in ridge_vertices[k]])

                    # Get a unit normal vector to the polygonal facet
                    r01 = vertex_coords[1] - vertex_coords[0]
                    r12 = vertex_coords[2] - vertex_coords[1]
                    norm_vec = np.cross(r01, r12)
                    norm_vec /= np.linalg.norm(norm_vec)

                    # Determine projection axis:
                    # c0 is the largest coordinate (x, y, or z) of the normal
                    # vector. We project along the c0 axis and use c1, c2 axes
                    # for computing the projected area.
                    c0 = np.argmax(np.abs(norm_vec))
                    c1 = (c0 + 1) % 3
                    c2 = (c0 + 2) % 3

                    vc1 = vertex_coords[:, c1]
                    vc2 = vertex_coords[:, c2]

                    # Use shoelace formula for the projected area
                    projected_area = 0.5*np.abs(
                        np.dot(vc1, np.roll(vc2, 1)) -
                        np.dot(vc2, np.roll(vc1, 1)))

                    # Project back to get the true area (which is the weight)
                    weight = projected_area / np.abs(norm_vec[c0])
            else:
                # This point was on the boundary, so as far as qhull
                # is concerned its ridge goes out to infinity
                weight = 0

            all_bonds[(index_i, index_j)] = max(all_bonds.get(
                                                (index_i, index_j), 0), weight)
            all_bonds[(index_j, index_i)] = max(all_bonds.get(
                                                (index_j, index_i), 0), weight)

        # Build neighbor list based on voronoi neighbors
        cdef np.ndarray[np.float32_t, ndim=1] weights = \
            np.asarray(list(all_bonds.values()), dtype=np.float32)
        cdef np.ndarray[np.uint64_t, ndim=2] bond_indices = \
            np.asarray(list(all_bonds.keys()), dtype=np.uint64)

        cdef np.ndarray[np.long_t, ndim=1] sort_indices = \
            np.lexsort((bond_indices[:, 1], bond_indices[:, 0]))

        bond_indices = bond_indices[sort_indices]
        weights = weights[sort_indices]

        self._nlist = freud.locality.NeighborList.from_arrays(
            np.max(bond_indices[:, 0])+1, np.max(bond_indices[:, 1])+1,
            bond_indices[:, 0], bond_indices[:, 1], weights=weights)
        return self

    @Compute._computed_method("computeNeighbors")
    def getNeighbors(self, numShells):
        R"""Get well-sorted neighbors from cumulative Voronoi shells for each
        particle by specifying :code:`numShells`.

        Must call :meth:`~.computeNeighbors()` before this method.

        Args:
            numShells (int): Number of neighbor shells.
        """
        nlist = self.nlist
        max_index = np.max(nlist[:]+1)
        # convert neighbor list to a csr matrix and set weights = 1
        sparse_neighbors = csr_matrix(
            (np.ones(len(nlist)), (nlist.index_i, nlist.index_j)),
            shape=(max_index, max_index))
        # take a numShell power of the matrix
        # sum over all shell neighbors, and convert to a linked list
        sparse_neighbors = sum(
            [sparse_neighbors**k for k in range(1, numShells+1)])
        lil_neighbors = sparse_neighbors.tolil()
        # extract all nonzero entries, ignoring diagonals
        nonzero_entries = lil_neighbors.nonzero()
        neighbors = [[] for _ in range(lil_neighbors.shape[0])]
        for i in range(nonzero_entries[0].shape[0]):
            if nonzero_entries[0][i] != nonzero_entries[1][i]:
                neighbors[nonzero_entries[0][i]].append(nonzero_entries[1][i])
        # return a list of list of well-sorted neighbors
        return neighbors

    @Compute._computed_property("computeNeighbors")
    def nlist(self):
        R"""Returns a neighbor list object.

        In the neighbor list, each neighbor pair has a weight value.

        In 2D systems, the bond weight is the "ridge length" of the Voronoi
        boundary line between the neighboring particles.

        In 3D systems, the bond weight is the "ridge area" of the Voronoi
        boundary polygon between the neighboring particles.

        Returns:
            :class:`~.locality.NeighborList`: Neighbor list.
        """
        return self._nlist

    @Compute._compute("computeVolumes")
    def computeVolumes(self):
        R"""Computes volumes (areas in 2D) of Voronoi cells.

        .. versionadded:: 0.8

        Must call :meth:`freud.voronoi.Voronoi.compute()` before this
        method. Retrieve the results with the `volumes` attribute.
        """
        polytope_verts = self.polytopes
        self._poly_volumes = np.zeros(shape=len(polytope_verts))

        for i, verts in enumerate(polytope_verts):
            is2D = np.all(self._poly_verts[0][:, -1] == 0)
            hull = ConvexHull(verts[:, :2 if is2D else 3])
            self._poly_volumes[i] = hull.volume

        return self

    @Compute._computed_property("computeVolumes")
    def volumes(self):
        R"""Returns an array of volumes (areas in 2D) corresponding to Voronoi
        cells.

        .. versionadded:: 0.8

        Must call :meth:`freud.voronoi.Voronoi.computeVolumes()` before this
        method.

        If the buffer width is too small, then some polytopes may not be
        closed (they may have a boundary at infinity), and these polytopes'
        volumes/areas are excluded from the list.

        The length of the list returned by this method should be the same
        as the array of positions used in the
        :meth:`freud.voronoi.Voronoi.compute()` method, if all the polytopes
        are closed. Otherwise try using a larger buffer width.

        Returns:
            (:math:`\left(N_{cells} \right)`) :class:`numpy.ndarray`:
                Voronoi polytope volumes/areas.
        """
        return self._poly_volumes

    def __repr__(self):
        return "freud.voronoi.{cls}(box={box}, buff={buff})".format(
            cls=type(self).__name__, box=self._box, buff=self._buff)

    def __str__(self):
        return repr(self)
