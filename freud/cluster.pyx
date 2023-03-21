# Copyright (c) 2010-2023 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

r"""
The :class:`freud.cluster` module aids in finding and computing the properties
of clusters of points in a system.
"""

from cython.operator cimport dereference

from freud.locality cimport _PairCompute
from freud.util cimport _Compute

import numpy as np

import freud.locality
import freud.util

cimport numpy as np

cimport freud._cluster
cimport freud.locality
cimport freud.util

# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef class Cluster(_PairCompute):
    """Finds clusters using a network of neighbors.

    Given a set of points and their neighbors, :class:`freud.cluster.Cluster`
    will determine all of the connected components of the network formed by
    those neighbor bonds. That is, two points are in the same cluster if and
    only if a path exists between them on the network of bonds. The class
    attribute :code:`cluster_idx` holds an array of cluster indices for each
    point. By the definition of a cluster, points that are not bonded to any
    other point end up in their own 1-point cluster.

    Identifying micelles is one use-case for finding clusters. This operation
    is somewhat different, though. In a cluster of points, each and every point
    belongs to one and only one cluster. However, because a string of points
    belongs to a polymer, that single polymer may be present in more than one
    cluster. To handle this situation, an optional layer is presented on top of
    the :code:`cluster_idx` array. Given a key value per point (e.g. the
    polymer id), the compute function will process clusters with the key values
    in mind and provide a list of keys that are present in each cluster in the
    attribute :code:`cluster_keys`, as a list of lists. If keys are not
    provided, every point is assigned a key corresponding to its index, and
    :code:`cluster_keys` contains the point ids present in each cluster.
    """

    cdef freud._cluster.Cluster * thisptr

    def __cinit__(self):
        self.thisptr = new freud._cluster.Cluster()

    def __init__(self):
        pass

    def __dealloc__(self):
        del self.thisptr

    def compute(self, system, keys=None, neighbors=None):
        r"""Compute the clusters for the given set of points.

        Args:
            system:
                Any object that is a valid argument to
                :class:`freud.locality.NeighborQuery.from_system`.
            keys ((:math:`N_{points}`) :class:`numpy.ndarray`):
                Membership keys, one for each point.
            neighbors (:class:`freud.locality.NeighborList` or dict, optional):
                Either a :class:`NeighborList <freud.locality.NeighborList>` of
                neighbor pairs to use in the calculation, or a dictionary of
                `query arguments
                <https://freud.readthedocs.io/en/stable/topics/querying.html>`_
                (Default value: None).
        """
        cdef:
            freud.locality.NeighborQuery nq
            freud.locality.NeighborList nlist
            freud.locality._QueryArgs qargs
            const float[:, ::1] l_query_points
            unsigned int num_query_points

        nq, nlist, qargs, l_query_points, num_query_points = \
            self._preprocess_arguments(system, neighbors=neighbors)

        cdef unsigned int* l_keys_ptr = NULL
        cdef unsigned int[::1] l_keys
        if keys is not None:
            l_keys = freud.util._convert_array(
                keys, shape=(num_query_points, ), dtype=np.uint32)
            l_keys_ptr = &l_keys[0]

        self.thisptr.compute(
            nq.get_ptr(),
            nlist.get_ptr(),
            dereference(qargs.thisptr),
            l_keys_ptr)
        return self

    @_Compute._computed_property
    def num_clusters(self):
        """int: The number of clusters."""
        return self.thisptr.getNumClusters()

    @_Compute._computed_property
    def cluster_idx(self):
        """(:math:`N_{points}`) :class:`numpy.ndarray`: The cluster index for
        each point."""
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getClusterIdx(),
            freud.util.arr_type_t.UNSIGNED_INT)

    @_Compute._computed_property
    def cluster_keys(self):
        """list(list): A list of lists of the keys contained in each
        cluster."""
        cluster_keys = self.thisptr.getClusterKeys()
        return cluster_keys

    def __repr__(self):
        return "freud.cluster.{cls}()".format(cls=type(self).__name__)

    def plot(self, ax=None):
        """Plot cluster distribution.

        Args:
            ax (:class:`matplotlib.axes.Axes`, optional): Axis to plot on. If
                :code:`None`, make a new figure and axis.
                (Default value = :code:`None`)

        Returns:
            (:class:`matplotlib.axes.Axes`): Axis with the plot.
        """
        import freud.plot
        try:
            values, counts = np.unique(self.cluster_idx, return_counts=True)
        except ValueError:
            return None
        return freud.plot.clusters_plot(
            values, counts, num_clusters_to_plot=10, ax=ax)

    def _repr_png_(self):
        try:
            import freud.plot
            return freud.plot._ax_to_bytes(self.plot())
        except (AttributeError, ImportError):
            return None


cdef class ClusterProperties(_Compute):
    r"""Routines for computing properties of point clusters.

    Given a set of points and cluster ids (from :class:`~.Cluster` or another
    source), this class determines the following properties for each cluster:

     - Geometric center
     - Center of mass
     - Gyration tensor
     - Moment of inertia tensor
     - Size (number of points)
     - Mass (total mass of each cluster)

    Note:
        The center of mass and geometric center for each cluster are computed
        using the minimum image convention
    """

    cdef freud._cluster.ClusterProperties * thisptr

    def __cinit__(self):
        self.thisptr = new freud._cluster.ClusterProperties()

    def __init__(self):
        pass

    def __dealloc__(self):
        del self.thisptr

    def compute(self, system, cluster_idx, masses=None):
        r"""Compute properties of the point clusters.
        Loops over all points in the given array and determines the geometric
        center, center of mass, moment of inertia, gyration tensors, and
        radius of gyration :cite:`Vymetal2011` of each cluster. After calling
        this method, properties can be accessed via their attributes.

        Example::

            >>> import freud
            >>> # Compute clusters using box, positions, and nlist data
            >>> box, points = freud.data.make_random_system(10, 100)
            >>> cl = freud.cluster.Cluster()
            >>> cl.compute((box, points), neighbors={'r_max': 1.0})
            freud.cluster.Cluster()
            >>> # Compute cluster properties based on identified clusters
            >>> cl_props = freud.cluster.ClusterProperties()
            >>> cl_props.compute((box, points), cl.cluster_idx)
            freud.cluster.ClusterProperties()

        Args:
            system:
                Any object that is a valid argument to
                :class:`freud.locality.NeighborQuery.from_system`.
            cluster_idx ((:math:`N_{points}`,) :class:`np.ndarray`):
                Cluster indexes for each point.
            masses ((:math:`N_{points}`, ) :class:`numpy.ndarray`):
                Masses corresponding to each point, defaulting to 1 if not
                provided or :code:`None` (Default value = :code:`None`).
        """
        cdef freud.locality.NeighborQuery nq = \
            freud.locality.NeighborQuery.from_system(system)
        cluster_idx = freud.util._convert_array(
            cluster_idx, shape=(nq.points.shape[0], ), dtype=np.uint32)
        cdef const unsigned int[::1] l_cluster_idx = cluster_idx

        cdef float* l_masses_ptr = NULL
        cdef float[::1] l_masses
        if masses is not None:
            l_masses = freud.util._convert_array(masses, shape=(len(masses), ))
            l_masses_ptr = &l_masses[0]

        self.thisptr.compute(nq.get_ptr(),
                             <unsigned int*> &l_cluster_idx[0],
                             l_masses_ptr)
        return self

    @_Compute._computed_property
    def centers(self):
        r"""(:math:`N_{clusters}`, 3) :class:`numpy.ndarray`: The geometric centers
        of the clusters, independent of mass and defined as

        .. math::

            \mathbf{C}_g^k = \frac{1}{N_k} \sum_{i=0}^{N_k} \mathbf{r_i}

        where :math:`\mathbf{C}_g^k` is the center of the :math:`k` th cluster,
        :math:`N_k` is the number of particles in the :math:`k` th cluster and
        :math:`\mathbf{r_i}` are their positions.
        """
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getClusterCenters(),
            freud.util.arr_type_t.FLOAT, 3)

    @_Compute._computed_property
    def centers_of_mass(self):
        r"""(:math:`N_{clusters}`, 3) :class:`numpy.ndarray`: The centers of mass
        of the clusters:

        .. math::

            \mathbf{C}_m^k = \frac{1}{M_k} \sum_{i=0}^{N_k} m_i \mathbf{r_i}

        where :math:`\mathbf{C}_m^k` is the center of mass of the :math:`k` th
        cluster, :math:`M_k` is the total mass of particles in the :math:`k` th
        cluster, :math:`\mathbf{r_i}` are their positions and :math:`m_i` are
        their masses.
        """
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getClusterCentersOfMass(),
            freud.util.arr_type_t.FLOAT, 3)

    @_Compute._computed_property
    def gyrations(self):
        r"""(:math:`N_{clusters}`, 3, 3) :class:`numpy.ndarray`: The gyration
        tensors of the clusters. Normalized by particle number:

        .. math::

            \mathbf{S}_k = \frac{1}{N_k} \begin{bmatrix}
            \sum_i x_i^2    & \sum_i x_i y_i  & \sum_i x_i z_i \\
            \sum_i y_i x_i  & \sum_i y_i^2    & \sum_i y_i z_i \\
            \sum_i z_i x_i  & \sum_i z_i y_i  & \sum_i z_i^2   \\
            \end{bmatrix}

        where :math:`\mathbf{S}_k` is the gyration tensor of the :math:`k` th
        cluster.
        """
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getClusterGyrations(),
            freud.util.arr_type_t.FLOAT)

    @_Compute._computed_property
    def inertia_tensors(self):
        r"""(:math:`N_{clusters}`, 3, 3) :class:`numpy.ndarray`: The inertia
        tensors of the clusters. Neither normalized by mass nor number:

        .. math::

            \mathbf{I}_k = \begin{bmatrix}
            \sum_i m_i(y_i^2+z_i^2)& \sum_i -m_i(x_iy_i)& \sum_i -m_i(x_iz_i)\\
            \sum_i -m_i(y_ix_i)& \sum_i m_i(x_i^2+z_i^2)& \sum_i -m_i(y_iz_i)\\
            \sum_i -m_i(z_ix_i)& \sum_i -m_i(z_iy_i)& \sum_i m_i(y_i^2+x_i^2)\\
            \end{bmatrix}

        where :math:`\mathbf{I}_k` is the inertia tensor of the :math:`k` th
        cluster.
        """
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getClusterMomentsOfInertia(),
            freud.util.arr_type_t.FLOAT)

    @_Compute._computed_property
    def sizes(self):
        """(:math:`N_{clusters}`) :class:`numpy.ndarray`: The cluster sizes."""
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getClusterSizes(),
            freud.util.arr_type_t.UNSIGNED_INT)

    @_Compute._computed_property
    def cluster_masses(self):
        """(:math:`N_{clusters}`) :class:`numpy.ndarray`: The total mass of
        particles in each cluster.
        """
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getClusterMasses(),
            freud.util.arr_type_t.FLOAT)

    @_Compute._computed_property
    def radii_of_gyration(self):
        r"""(:math:`N_{clusters}`,) :class:`numpy.ndarray`: The radius of
        gyration of each cluster. Defined by IUPAP as

        .. math::

            R_g^k = \left(\frac{1}{M} \sum_{i=0}^{N_k} m_i s_i^2 \right)^{1/2}

        where :math:`s_i` is the distance of particle :math:`i` from
        the center of mass.

        """
        return np.sqrt(np.trace(self.inertia_tensors, axis1=-2, axis2=-1)
                       /(2*self.cluster_masses))

    def __repr__(self):
        return "freud.cluster.{cls}()".format(cls=type(self).__name__)
