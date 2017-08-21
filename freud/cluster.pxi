# Copyright (c) 2010-2016 The Regents of the University of Michigan
# This file is part of the Freud project, released under the BSD 3-Clause License.

from freud.util._VectorMath cimport vec3
cimport freud._cluster as cluster
cimport freud._box as _box
import numpy as np
cimport numpy as np
import freud.common
from libcpp.vector cimport vector

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef class Cluster:
    """Finds clusters in a set of points.

    Given a set of coordinates and a cutoff, Cluster will determine all of the clusters of points that are made
    up of points that are closer than the cutoff. Clusters are labelled from 0 to the number of clusters-1
    and an index array is returned where cluster_idx[i] is the cluster index in which particle i is found.
    By the definition of a cluster, points that are not within the cutoff of another point end up in their own
    1-particle cluster.

    Identifying micelles is one primary use-case for finding clusters. This operation is somewhat different, though.
    In a cluster of points, each and every point belongs to one and only one cluster. However, because a string
    of points belongs to a polymer, that single polymer may be present in more than one cluster. To handle this
    situation, an optional layer is presented on top of the cluster_idx array. Given a key value per particle
    (i.e. the polymer id), the computeClusterMembership function will process cluster_idx with the key values in mind
    and provide a list of keys that are present in each cluster.

    .. moduleauthor:: Joshua Anderson <joaander@umich.edu>

    :param box: :py:class:`freud.box.Box`
    :param rcut: Particle distance cutoff
    :type box: :py:class:`freud.box.Box`
    :type rcut: float

    .. note::
        2D: Cluster properly handles 2D boxes. As with everything else in freud, 2D points must be passed in as \
        3 component vectors :math:`\\left(x,y,0\\right)`. Failing to set 0 in the third component will lead to undefined \
        behavior.
    """
    cdef cluster.Cluster *thisptr
    cdef box
    cdef rmax

    def __cinit__(self, box, float rcut):
        cdef _box.Box cBox = _box.Box(box.getLx(), box.getLy(), box.getLz(),
            box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr = new cluster.Cluster(cBox, rcut)
        self.box = box
        self.rmax = rcut

    def __dealloc__(self):
        del self.thisptr

    @property
    def box(self):
        """Return the stored Freud Box

        :return: Freud Box
        :rtype: :py:class:`freud.box.Box`
        """
        return self.getBox()

    def getBox(self):
        """Return the stored Freud Box

        :return: Freud Box
        :rtype: :py:class:`freud.box.Box`
        """
        return BoxFromCPP(self.thisptr.getBox())

    def computeClusters(self, points, nlist=None):
        """Compute the clusters for the given set of points

        :param points: particle coordinates
        :param nlist: :py:class:`freud.locality.NeighborList` object to use to find bonds
        :type points: :class:`numpy.ndarray`, shape=(:math:`N_{particles}`, 3), dtype= :class:`numpy.float32`
        :type nlist: :py:class:`freud.locality.NeighborList`
        """
        points = freud.common.convert_array(points, 2, dtype=np.float32, contiguous=True)
        if points.shape[1] != 3:
            raise RuntimeError('Need a list of 3D points for computeClusters()')

        defaulted_nlist = make_default_nlist(self.box, points, points, self.rmax, nlist, True)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        cdef locality.NeighborList *nlist_ptr = nlist_.get_ptr()

        cdef np.ndarray cPoints = points
        cdef unsigned int Np = points.shape[0]
        with nogil:
            self.thisptr.computeClusters(nlist_ptr, <vec3[float]*> cPoints.data, Np)
        return self

    def computeClusterMembership(self, keys):
        """Compute the clusters with key membership

        Loops overa all particles and adds them to a list of sets.
        Each set contains all the keys that are part of that cluster.

        Get the computed list with getClusterKeys().

        :param keys: Membership keys, one for each particle
        :type keys: :class:`numpy.ndarray`, shape=(:math:`N_{particles}`), dtype= :class:`numpy.uint32`
        """
        keys = freud.common.convert_array(keys, 1, dtype=np.uint32, contiguous=True)
        N = self.getNumParticles()
        if keys.shape[0] != N:
            raise RuntimeError('keys must be a 1D array of length NumParticles')
        cdef np.ndarray cKeys = keys
        with nogil:
            self.thisptr.computeClusterMembership(<unsigned int *>cKeys.data)
        return self

    @property
    def num_clusters(self):
        """Returns the number of clusters

        :return: number of clusters
        :rtype: int
        """
        return self.getNumClusters()

    def getNumClusters(self):
        """Returns the number of clusters

        :return: number of clusters
        :rtype: int
        """
        return self.thisptr.getNumClusters()

    @property
    def num_particles(self):
        """Returns the number of particles

        :return: number of particles
        :rtype: int
        """
        return self.getNumParticles()

    def getNumParticles(self):
        """Returns the number of particles
        :return: number of particles
        :rtype: int
        """
        return self.thisptr.getNumParticles()

    @property
    def cluster_idx(self):
        """Returns 1D array of Cluster idx for each particle

        :return: 1D array of cluster idx
        :rtype: :class:`numpy.ndarray`, shape=(:math:`N_{particles}`), dtype= :class:`numpy.uint32`
        """
        return self.getClusterIdx()

    def getClusterIdx(self):
        """
        Returns 1D array of Cluster idx for each particle

        :return: 1D array of cluster idx
        :rtype: :class:`numpy.ndarray`, shape=(:math:`N_{particles}`), dtype= :class:`numpy.uint32`
        """
        cdef unsigned int *cluster_idx_raw = self.thisptr.getClusterIdx().get()
        cdef np.npy_intp nP[1]
        nP[0] = <np.npy_intp>self.thisptr.getNumParticles()
        cdef np.ndarray[np.uint32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nP, np.NPY_UINT32, <void*>cluster_idx_raw)
        return result

    @property
    def cluster_keys(self):
        """Returns the keys contained in each cluster

        :return: list of lists of each key containted in clusters
        :rtype: list

        .. todo: Determine correct way to export. As-is, I do not particularly like how it was previously handled.
        """
        return self.getClusterIdx()

    def getClusterKeys(self):
        """Returns the keys contained in each cluster

        :return: list of lists of each key containted in clusters
        :rtype: list

        .. todo: Determine correct way to export. As-is, I do not particularly like how it was previously handled.
        """
        cluster_keys = self.thisptr.getClusterKeys()
        return cluster_keys


cdef class ClusterProperties:
    """Routines for computing properties of point clusters

    Given a set of points and cluster_idx (from :class:`~.Cluster`, or another source), ClusterProperties determines
    the following properties for each cluster:

     - Center of mass
     - Gyration radius tensor

    m_cluster_com stores the computed center of mass for each cluster (properly handling periodic boundary conditions,
    of course) as a :class:`numpy.ndarray`, shape= :math:`\\left(N_{clusters}, 3 \\right)`.

    m_cluster_G stores a :math:`3 \\times 3` G tensor for each cluster. Index cluster c, element j, i with the following:
    m_cluster_G[c*9 + j*3 + i]. The tensor is symmetric, so the choice of i and j are irrelevant. This is passed
    back to python as a :math:`N_{clusters} \\times 3 \\times 3` numpy array.

    .. moduleauthor:: Joshua Anderson <joaander@umich.edu>

    :param box: simulation box
    :type box: :py:class:`freud.box.Box`
    """
    cdef cluster.ClusterProperties *thisptr

    def __cinit__(self, box):
        cdef _box.Box cBox = _box.Box(box.getLx(), box.getLy(), box.getLz(),
            box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr = new cluster.ClusterProperties(cBox)


    def __dealloc__(self):
        del self.thisptr


    @property
    def box(self):
        """Return the stored Freud Box

        :return: Freud Box
        :rtype: :py:class:`freud.box.Box`
        """
        return self.getBox()

    def getBox(self):
        """Return the stored :py:class:`freud.box.Box` object

        :return: Freud Box
        :rtype: :py:class:`freud.box.Box`
        """
        return BoxFromCPP(self.thisptr.getBox())

    def computeProperties(self, points, cluster_idx):
        """Compute properties of the point clusters

        Loops over all points in the given array and determines the center of mass of the cluster
        as well as the G tensor. These can be accessed after the call to compute with :meth:`~.getClusterCOM()` and \
        :meth:`~.getClusterG()`.

        :param points: Positions of the particles making up the clusters
        :param cluster_idx: Index of which cluster each point belongs to
        :type points: :class:`numpy.ndarray`, shape=(:math:`N_{particles}`, 3), dtype= :class:`numpy.float32`
        :type cluster_idx: :class:`numpy.ndarray`, shape=(:math:`N_{particles}`), dtype= :class:`numpy.uint32`
        """
        points = freud.common.convert_array(points, 2, dtype=np.float32, contiguous=True)
        if points.shape[1] != 3:
            raise RuntimeError('Need a list of 3D points for computeClusterProperties()')
        cluster_idx = freud.common.convert_array(cluster_idx, 1, dtype=np.uint32, contiguous=True)
        if cluster_idx.shape[0] != points.shape[0]:
            raise RuntimeError('cluster_idx must be a 1D array of matching length/number of particles to points')
        cdef np.ndarray cPoints = points
        cdef np.ndarray cCluster_idx= cluster_idx
        cdef unsigned int Np = points.shape[0]
        with nogil:
            self.thisptr.computeProperties(<vec3[float]*> cPoints.data, <unsigned int *> cCluster_idx.data, Np)
        return self

    @property
    def num_clusters(self):
        """Returns the number of clusters

        :return: number of clusters
        :rtype: int
        """
        return self.getNumClusters()

    def getNumClusters(self):
        """Count the number of clusters found in the last call to :meth:`~.computeProperties()`

        :return: number of clusters
        :rtype: int
        """
        return self.thisptr.getNumClusters()

    @property
    def cluster_COM(self):
        """Returns the cluster center of mass the last computed cluster_com

        :return: numpy array of cluster center of mass coordinates :math:`\\left(x,y,z\\right)`
        :rtype: :class:`numpy.ndarray`, shape=(:math:`N_{clusters}`, 3), dtype= :class:`numpy.float32`
        """
        return self.getClusterCOM()

    def getClusterCOM(self):
        """Returns the cluster center of mass the last computed cluster_com

        :return: numpy array of cluster center of mass coordinates :math:`\\left(x,y,z\\right)`
        :rtype: :class:`numpy.ndarray`, shape=(:math:`N_{clusters}`, 3), dtype= :class:`numpy.float32`
        """
        cdef vec3[float] *cluster_com_raw = self.thisptr.getClusterCOM().get()
        cdef np.npy_intp nClusters[2]
        nClusters[0] = <np.npy_intp>self.thisptr.getNumClusters()
        nClusters[1] = 3
        cdef np.ndarray[np.float32_t, ndim=2] result = np.PyArray_SimpleNewFromData(2, nClusters, np.NPY_FLOAT32, <void*>cluster_com_raw)
        return result

    @property
    def cluster_G(self):
        """Returns the cluster G tensors computed by the last call to computeProperties

        :return: numpy array of cluster center of mass coordinates :math:`\\left(x,y,z\\right)`
        :rtype: :class:`numpy.ndarray`, shape=(:math:`N_{clusters}`, 3, 3), dtype= :class:`numpy.float32`
        """
        return self.getClusterG()

    def getClusterG(self):
        """Returns the cluster G tensors computed by the last call to computeProperties

        :return: numpy array of cluster center of mass coordinates :math:`\\left(x,y,z\\right)`
        :rtype: :class:`numpy.ndarray`, shape=(:math:`N_{clusters}`, 3, 3), dtype= :class:`numpy.float32`
        """
        cdef float *cluster_G_raw = self.thisptr.getClusterG().get()
        cdef np.npy_intp nClusters[3]
        nClusters[0] = <np.npy_intp>self.thisptr.getNumClusters()
        nClusters[1] = 3
        nClusters[2] = 3
        cdef np.ndarray[np.float32_t, ndim=3] result = np.PyArray_SimpleNewFromData(3, nClusters, np.NPY_FLOAT32, <void*>cluster_G_raw)
        return result

    @property
    def cluster_sizes(self):
        """Returns the cluster sizes computed by the last call to computeProperties

        :return: numpy array of sizes of each cluster
        :rtype: :class:`numpy.ndarray`, shape=(:math:`N_{clusters}`), dtype= :class:`numpy.uint32`
        """
        return self.getClusterSizes()

    def getClusterSizes(self):
        """Returns the cluster sizes computed by the last call to computeProperties

        :return: numpy array of sizes of each cluster
        :rtype: :class:`numpy.ndarray`, shape=(:math:`N_{clusters}`), dtype= :class:`numpy.uint32`
        """
        cdef unsigned int *cluster_sizes_raw = self.thisptr.getClusterSize().get()
        cdef np.npy_intp nClusters[1]
        nClusters[0] = <np.npy_intp>self.thisptr.getNumClusters()
        cdef np.ndarray[np.uint32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nClusters, np.NPY_UINT32, <void*>cluster_sizes_raw)
        return result
