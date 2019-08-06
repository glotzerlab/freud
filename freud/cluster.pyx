# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

R"""
The :class:`freud.cluster` module aids in finding and computing the properties
of clusters of points in a system.
"""

import numpy as np
import warnings
import sys
import freud.common
import freud.locality

from cython.operator cimport dereference
from freud.common cimport Compute
from freud.util cimport vec3, uint
from functools import wraps

cimport freud._cluster
cimport freud.box, freud.locality
cimport freud.util

cimport numpy as np

# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

def resolve_arrays(array_names):
    """Decorator that ensures that all ManagedArrays are released to C++ if
    possible to minimize memory reallocations.

    Args:
        array_names (str or list(str)): ManagedArray attributes that should be
                                        released and reacquired.
    Returns:
        callable: A function that behaves as a decorator for a compute call.
    """
    if type(array_names) is str:
        array_names = [array_names]

    def wrapper(func):
        """The wrapper is the actual decorator that is called on the compute
        function.

        Args:
            func (callable): The compute function to manage arrays for.

        Returns:
            callable: A compute function that manages arrays.
        """
        @wraps(func)
        def acquire_and_compute(self, *args, **kwargs):
            """This function is the replacement for compute.

            Args:
                *args: Any positional arguments to the compute call.
                **kwargs: Any keyword arguments to the compute call.

            Returns:
                callable: A compute function that manages arrays.
            """
            # If other objects (e.g. NumPy arrays) are referencing this one, then
            # we reallocate a new Python wrapper object. Otherwise, we relinquish
            # control of the underlying array to the C++ class for its computation.
            # In either case, the Python wrapper class reacquires ownership at the
            # end.
            cdef freud.util.ManagedArrayWrapper array
            cdef freud.util.arr_ptr_t managed_array
            for array_name in array_names:
                refcount = sys.getrefcount(getattr(self, array_name))
                array = <freud.util.ManagedArrayWrapper> getattr(self, array_name)
                if refcount <= 2:
                    array.release()
                else:
                    managed_array.uint_ptr = array.sourceptr.uint_ptr
                    setattr(self, array_name,
                            freud.util.ManagedArrayWrapper.init(managed_array, np.NPY_UINT32))
            ret_val = func(self, *args, **kwargs)

            # Store the array locally again.
            for array_name in array_names:
                array = <freud.util.ManagedArrayWrapper> getattr(self, array_name)
                array.acquire()

            return ret_val
        return acquire_and_compute
    return wrapper


cdef class Cluster(Compute):
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
        r_max (float):
            Particle distance cutoff.

    .. note::
        **2D:** :class:`freud.cluster.Cluster` properly handles 2D boxes.
        The points must be passed in as :code:`[x, y, 0]`.
        Failing to set z=0 will lead to undefined behavior.

    Attributes:
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
    cdef float r_max
    cdef public freud.util.ManagedArrayWrapper __cluster_idx

    def __cinit__(self, float r_max):
        self.thisptr = new freud._cluster.Cluster(r_max)
        self.r_max = r_max
        cdef freud.util.arr_ptr_t managed_array
        managed_array.uint_ptr = &self.thisptr.getClusterIdx()
        self.__cluster_idx = freud.util.ManagedArrayWrapper.init(managed_array, np.NPY_UINT32)
        self.__cluster_idx.acquire()

    def __dealloc__(self):
        del self.thisptr

    @resolve_arrays('__cluster_idx')
    @Compute._compute("compute")
    def compute(self, box, points, nlist=None):
        R"""Compute the clusters for the given set of points.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            points ((:math:`N_{particles}`, 3) :class:`np.ndarray`):
                Particle coordinates.
            nlist (:class:`freud.locality.NeighborList`, optional):
                Object to use to find bonds (Default value = :code:`None`).
        """
        cdef freud.box.Box b = freud.common.convert_box(box)

        nq_nlist = freud.locality.make_nq_nlist(b, points, nlist)
        cdef freud.locality.NeighborQuery nq = nq_nlist[0]
        cdef freud.locality.NlistptrWrapper nlistptr = nq_nlist[1]

        cdef freud.locality._QueryArgs qargs = freud.locality._QueryArgs(
            mode="ball", r_max=self.r_max, exclude_ii=True)
        points = nq.points

        cdef const float[:, ::1] l_points = points
        cdef unsigned int Np = l_points.shape[0]
        self.__cluster_idx.set_shape((Np, ))
        cdef freud.util.ManagedArray[vec3[float]] arr = freud.util.ManagedArray[vec3[float]](<vec3[float] *> &l_points[0, 0], Np)
        with nogil:
            self.thisptr.compute(
                nq.get_ptr(),
                nlistptr.get_ptr(),
                arr, dereference(qargs.thisptr))

        return self

    @Compute._compute("computeClusterMembership")
    def computeClusterMembership(self, keys):
        R"""Compute the clusters with key membership.
        Loops over all particles and adds them to a list of sets.
        Each set contains all the keys that are part of that cluster.
        Get the computed list with :attr:`~cluster_keys`.

        Args:
            keys ((:math:`N_{particles}`) :class:`numpy.ndarray`):
                Membership keys, one for each particle.
        """
        keys = freud.common.convert_array(
            keys, shape=(self.num_particles, ), dtype=np.uint32)
        cdef const unsigned int[::1] l_keys = keys
        with nogil:
            self.thisptr.computeClusterMembership(<unsigned int*> &l_keys[0])
        return self

    @Compute._computed_property("compute")
    def num_clusters(self):
        return self.thisptr.getNumClusters()

    @Compute._computed_property("compute")
    def num_particles(self):
        return self.thisptr.getNumParticles()

    @Compute._computed_property("compute")
    def cluster_idx(self):
        return np.asarray(self.__cluster_idx)

    @Compute._computed_property("computeClusterMembership")
    def cluster_keys(self):
        cluster_keys = self.thisptr.getClusterKeys()
        return cluster_keys

    def __repr__(self):
        return ("freud.cluster.{cls}("
                "r_max={r_max})").format(cls=type(self).__name__,
                                         r_max=self.r_max)

    def __str__(self):
        return repr(self)

    @Compute._computed_method("compute")
    def plot(self, ax=None):
        """Plot cluster distribution.

        Args:
            ax (:class:`matplotlib.axes.Axes`, optional): Axis to plot on. If
                :code:`None`, make a new figure and axis.
                (Default value = :code:`None`)

        Returns:
            (:class:`matplotlib.axes.Axes`): Axis with the plot.
        """
        import plot
        try:
            count = np.unique(self.cluster_idx, return_counts=True)
        except ValueError:
            return None
        else:
            return plot.plot_clusters(count[0], count[1],
                                      num_cluster_to_plot=10, ax=ax)

    def _repr_png_(self):
        import plot
        try:
            return plot.ax_to_bytes(self.plot())
        except AttributeError:
            return None


cdef class ClusterProperties(Compute):
    R"""Routines for computing properties of point clusters.

    Given a set of points and cluster ids (from :class:`~.Cluster`, or another
    source), ClusterProperties determines the following properties for each
    cluster:

     - Center of mass
     - Gyration tensor

    The computed center of mass for each cluster (properly handling periodic
    boundary conditions) can be accessed with :code:`cluster_COM` attribute.
    The :math:`3 \times 3` gyration tensor :math:`G` can be accessed with
    :code:`cluster_G` attribute.
    The tensor is symmetric for each cluster.

    .. moduleauthor:: Joshua Anderson <joaander@umich.edu>

    Attributes:
        num_clusters (int):
            The number of clusters.
        cluster_COM ((:math:`N_{clusters}`, 3) :class:`numpy.ndarray`):
            The center of mass of the last computed cluster.
        cluster_G ((:math:`N_{clusters}`, 3, 3) :class:`numpy.ndarray`):
            The cluster :math:`G` tensors computed by the last call to
            :meth:`~.compute()`.
        cluster_sizes ((:math:`N_{clusters}`) :class:`numpy.ndarray`):
            The cluster sizes computed by the last call to
            :meth:`~.compute()`.
    """
    cdef freud._cluster.ClusterProperties * thisptr

    def __cinit__(self):
        self.thisptr = new freud._cluster.ClusterProperties()

    def __dealloc__(self):
        del self.thisptr

    @Compute._compute("compute")
    def compute(self, box, points, cluster_idx):
        R"""Compute properties of the point clusters.
        Loops over all points in the given array and determines the center of
        mass of the cluster as well as the :math:`G` tensor. These can be
        accessed after the call to :meth:`~.compute()` with the
        :code:`cluster_COM` and :code:`cluster_G` attributes.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            points ((:math:`N_{particles}`, 3) :class:`np.ndarray`):
                Positions of the particles making up the clusters.
            cluster_idx (:class:`np.ndarray`):
                List of cluster indexes for each particle.
        """
        cdef freud.box.Box b = freud.common.convert_box(box)

        points = freud.common.convert_array(points, shape=(None, 3))
        cluster_idx = freud.common.convert_array(
            cluster_idx, shape=(points.shape[0], ), dtype=np.uint32)
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

    @Compute._computed_property("compute")
    def num_clusters(self):
        return self.thisptr.getNumClusters()

    @Compute._computed_property("compute")
    def cluster_COM(self):
        cdef unsigned int n_clusters = self.thisptr.getNumClusters()
        if not n_clusters:
            return np.asarray([[]], dtype=np.float32)
        cdef const float[:, ::1] cluster_COM = \
            <float[:n_clusters, :3]> (
                <float*> self.thisptr.getClusterCOM().get())
        return np.asarray(cluster_COM)

    @Compute._computed_property("compute")
    def cluster_G(self):
        cdef unsigned int n_clusters = self.thisptr.getNumClusters()
        if not n_clusters:
            return np.asarray([[[]]], dtype=np.float32)
        cdef const float[:, :, ::1] cluster_G = \
            <float[:n_clusters, :3, :3]> (
                <float*> self.thisptr.getClusterG().get())
        return np.asarray(cluster_G)

    @Compute._computed_property("compute")
    def cluster_sizes(self):
        cdef unsigned int n_clusters = self.thisptr.getNumClusters()
        if not n_clusters:
            return np.asarray([], dtype=np.uint32)
        cdef const unsigned int[::1] cluster_sizes = \
            <unsigned int[:n_clusters]> self.thisptr.getClusterSize().get()
        return np.asarray(cluster_sizes, dtype=np.uint32)

    def __repr__(self):
        return ("freud.cluster.{cls}()").format(
            cls=type(self).__name__)

    def __str__(self):
        return repr(self)
