# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

R"""
The :class:`freud.cluster` module aids in finding and computing the properties
of clusters of points in a system.
"""

import numpy as np
import warnings
import freud.common
import freud.locality

from cython.operator cimport dereference
from freud.util._VectorMath cimport vec3

cimport freud._cluster
cimport freud.box, freud.locality

cimport numpy as np

# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef class Cluster:
    R"""Finds clusters in a set of points.

    Given a set of coordinates and a cutoff, :class:`freud.cluster.Cluster`
    will determine all of the clusters of points that are made up of points
    that are closer than the cutoff. Clusters are 0-indexed. The class contains
    an index array, the :code:`cluster_idx` attribute, which can be used to
    identify which cluster a particle is associated with:
    :code:`cluster_obj.cluster_idx[i]` is the cluster index in which particle
    :code:`i` is found. By the definition of a cluster, points that are not
    within the cutoff of another point end up in their own 1-particle cluster.

    Identifying micelles is one primary use-case for finding clusters. This
    operation is somewhat different, though. In a cluster of points, each and
    every point belongs to one and only one cluster. However, because a string
    of points belongs to a polymer, that single polymer may be present in more
    than one cluster. To handle this situation, an optional layer is presented
    on top of the :code:`cluster_idx` array. Given a key value per particle
    (i.e. the polymer id), the computeClusterMembership function will process
    :code:`cluster_idx` with the key values in mind and provide a list of keys
    that are present in each cluster.

    .. moduleauthor:: Joshua Anderson <joaander@umich.edu>

    Args:
        box (:class:`freud.box.Box`):
            The simulation box.
        rcut (float):
            Particle distance cutoff.

    .. note::
        **2D:** :class:`freud.cluster.Cluster` properly handles 2D boxes.
        The points must be passed in as :code:`[x, y, 0]`.
        Failing to set z=0 will lead to undefined behavior.

    Attributes:
        box (:class:`freud.box.Box`):
            Box used in the calculation.
        num_clusters (int):
            The number of clusters.
        num_particles (int):
            The number of particles.
        cluster_idx ((:math:`N_{particles}`) :class:`numpy.ndarray`):
            The cluster index for each particle.
        cluster_keys (list(list)):
            A list of lists of the keys contained in each cluster.
    """
    cdef freud._cluster.Cluster * thisptr
    cdef freud.box.Box m_box
    cdef rmax

    def __cinit__(self, box, float rcut):
        cdef freud.box.Box b = freud.common.convert_box(box)
        self.thisptr = new freud._cluster.Cluster(rcut)
        self.m_box = b
        self.rmax = rcut

    def __dealloc__(self):
        del self.thisptr

    @property
    def box(self):
        return self.m_box

    def computeClusters(self, points, nlist=None, box=None):
        R"""Compute the clusters for the given set of points.

        Args:
            points ((:math:`N_{particles}`, 3) :class:`np.ndarray`):
                Particle coordinates.
            nlist (:class:`freud.locality.NeighborList`, optional):
                Object to use to find bonds (Default value = None).
            box (:class:`freud.box.Box`, optional):
                Simulation box (Default value = None).
        """

        # defaulted_nlist = freud.locality.make_default_nlist(
        #     self.m_box, points, points, self.rmax, nlist, True)
        # cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]

        cdef freud.locality.NeighborQuery nq = \
            freud.locality.make_default_nq(self.m_box, points)
        cdef freud._locality.NeighborList * nlistptr \
            = freud.locality.make_nlistptr(nlist)

        if isinstance(points, freud.locality.AABBQuery) \
                or isinstance(points, freud.locality.LinkCell):
            points = points.points

        points = freud.common.convert_array(
            points, 2, dtype=np.float32, contiguous=True)
        if points.shape[1] != 3:
            raise RuntimeError(
                'Need a list of 3D points for computeClusters()')

        cdef freud.box.Box b
        if box is None:
            b = self.m_box
        else:
            b = freud.common.convert_box(box)

        cdef const float[:, ::1] l_points = points
        cdef unsigned int Np = l_points.shape[0]
        with nogil:
            self.thisptr.computeClusters(
                nq.get_ptr(),
                dereference(b.thisptr), nlistptr,
                <vec3[float]*> &l_points[0, 0], Np)
        return self

    def computeClusterMembership(self, keys):
        R"""Compute the clusters with key membership.
        Loops over all particles and adds them to a list of sets.
        Each set contains all the keys that are part of that cluster.
        Get the computed list with :attr:`~cluster_keys`.

        Args:
            keys((:math:`N_{particles}`) :class:`numpy.ndarray`):
                Membership keys, one for each particle.
        """
        keys = freud.common.convert_array(
            keys, 1, dtype=np.uint32, contiguous=True)
        if keys.shape[0] != self.num_particles:
            raise RuntimeError(
                'keys must be a 1D array of length num_particles')
        cdef const unsigned int[::1] l_keys = keys
        with nogil:
            self.thisptr.computeClusterMembership(<unsigned int*> &l_keys[0])
        return self

    @property
    def num_clusters(self):
        return self.thisptr.getNumClusters()

    @property
    def num_particles(self):
        return self.thisptr.getNumParticles()

    @property
    def cluster_idx(self):
        cdef unsigned int n_particles = self.thisptr.getNumParticles()
        cdef const unsigned int[::1] cluster_idx = \
            <unsigned int[:n_particles]> self.thisptr.getClusterIdx().get()
        return np.asarray(cluster_idx)

    @property
    def cluster_keys(self):
        cluster_keys = self.thisptr.getClusterKeys()
        return cluster_keys

    def __repr__(self):
        return ("freud.cluster.{cls}(box={box}, " +
                "rcut={rcut})").format(cls=type(self).__name__,
                                       box=self.m_box.__repr__(),
                                       rcut=self.rmax)

    def __str__(self):
        return repr(self)


cdef class ClusterProperties:
    R"""Routines for computing properties of point clusters.

    Given a set of points and cluster ids (from :class:`~.Cluster`, or another
    source), ClusterProperties determines the following properties for each
    cluster:

     - Center of mass
     - Gyration tensor

    The computed center of mass for each cluster (properly handling periodic
    boundary conditions) can be accessed with :meth:`~.getClusterCOM()`.
    This returns a :math:`\left(N_{clusters}, 3 \right)`
    :class:`numpy.ndarray`.

    The :math:`3 \times 3` gyration tensor :math:`G` can be accessed with
    :meth:`~.getClusterG()`. This returns a :class:`numpy.ndarray`,
    shape= :math:`\left(N_{clusters} \times 3 \times 3\right)`.
    The tensor is symmetric for each cluster.

    .. moduleauthor:: Joshua Anderson <joaander@umich.edu>

    Args:
        box (:class:`freud.box.Box`): Simulation box.

    Attributes:
        box (:class:`freud.box.Box`):
            Box used in the calculation.
        num_clusters (int):
            The number of clusters.
        cluster_COM ((:math:`N_{clusters}`, 3) :class:`numpy.ndarray`):
            The center of mass of the last computed cluster.
        cluster_G ((:math:`N_{clusters}`, 3, 3) :class:`numpy.ndarray`):
            The cluster :math:`G` tensors computed by the last call to
            :meth:`~.computeProperties()`.
        cluster_sizes ((:math:`N_{clusters}`) :class:`numpy.ndarray`):
            The cluster sizes computed by the last call to
            :meth:`~.computeProperties()`.
    """
    cdef freud._cluster.ClusterProperties * thisptr
    cdef freud.box.Box m_box

    def __cinit__(self, box):
        cdef freud.box.Box b = freud.common.convert_box(box)
        self.thisptr = new freud._cluster.ClusterProperties()
        self.m_box = b

    def __dealloc__(self):
        del self.thisptr

    @property
    def box(self):
        return self.m_box

    def computeProperties(self, points, cluster_idx, box=None):
        R"""Compute properties of the point clusters.
        Loops over all points in the given array and determines the center of
        mass of the cluster as well as the :math:`G` tensor. These can be
        accessed after the call to :meth:`~.computeProperties()` with
        :meth:`~.getClusterCOM()` and :meth:`~.getClusterG()`.

        Args:
            points ((:math:`N_{particles}`, 3) :class:`np.ndarray`):
                Positions of the particles making up the clusters.
            cluster_idx (:class:`np.ndarray`):
                List of cluster indexes for each particle.
            box (:class:`freud.box.Box`, optional):
                Simulation box (Default value = None).
        """
        cdef freud.box.Box b
        if box is None:
            b = self.m_box
        else:
            b = freud.common.convert_box(box)

        points = freud.common.convert_array(
            points, 2, dtype=np.float32, contiguous=True)
        if points.shape[1] != 3:
            raise RuntimeError(
                'Need a list of 3D points for computeClusterProperties()')
        cluster_idx = freud.common.convert_array(
            cluster_idx, 1, dtype=np.uint32, contiguous=True)
        if cluster_idx.shape[0] != points.shape[0]:
            raise RuntimeError(
                ('cluster_idx must be a 1D array of matching length/number'
                    'of particles to points'))
        cdef const float[:, ::1] l_points = points
        cdef const unsigned int[::1] l_cluster_idx = cluster_idx
        cdef unsigned int Np = l_points.shape[0]
        with nogil:
            self.thisptr.computeProperties(
                dereference(b.thisptr),
                <vec3[float]*> &l_points[0, 0],
                <unsigned int*> &l_cluster_idx[0],
                Np)
        return self

    @property
    def num_clusters(self):
        return self.thisptr.getNumClusters()

    @property
    def cluster_COM(self):
        cdef unsigned int n_clusters = self.thisptr.getNumClusters()
        if not n_clusters:
            return np.asarray([[]], dtype=np.float32)
        cdef const float[:, ::1] cluster_COM = \
            <float[:n_clusters, :3]> (
                <float*> self.thisptr.getClusterCOM().get())
        return np.asarray(cluster_COM)

    @property
    def cluster_G(self):
        cdef unsigned int n_clusters = self.thisptr.getNumClusters()
        if not n_clusters:
            return np.asarray([[[]]], dtype=np.float32)
        cdef const float[:, :, ::1] cluster_G = \
            <float[:n_clusters, :3, :3]> (
                <float*> self.thisptr.getClusterG().get())
        return np.asarray(cluster_G)

    @property
    def cluster_sizes(self):
        cdef unsigned int n_clusters = self.thisptr.getNumClusters()
        if not n_clusters:
            return np.asarray([], dtype=np.uint32)
        cdef const unsigned int[::1] cluster_sizes = \
            <unsigned int[:n_clusters]> self.thisptr.getClusterSize().get()
        return np.asarray(cluster_sizes, dtype=np.uint32)

    def __repr__(self):
        return ("freud.cluster.{cls}(box={box})").format(
            cls=type(self).__name__, box=self.m_box.__repr__())

    def __str__(self):
        return repr(self)
