# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is part of the freud project, released under the BSD 3-Clause License.

from freud.util._VectorMath cimport vec3, quat
from libcpp.map cimport map
import numpy as np

cimport freud._box as _box
cimport freud._bond as bond
cimport numpy as np

# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef class BondingAnalysis:
    """Analyze the bond lifetimes and flux present in the system.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    :param num_particles: number of particles over which to calculate bonds
    :param num_bonds: number of bonds to track
    :type num_particles: unsigned int
    :type num bonds: unsigned int
    """
    cdef bond.BondingAnalysis * thisptr
    cdef unsigned int num_particles
    cdef unsigned int num_bonds

    def __cinit__(self, int num_particles, int num_bonds):
        self.num_particles = num_particles
        self.num_bonds = num_bonds
        self.thisptr = new bond.BondingAnalysis(num_particles, num_bonds)

    def __dealloc__(self):
        del self.thisptr

    def initialize(self, frame_0):
        """
        Calculates the changes in bonding states from one frame to the next.

        :param frame_0: first bonding frame (as output from
                        :py:class:`~.BondingR12` modules)
        :type frame_0: :class:`numpy.ndarray`,
                        shape=(:math:`N_{particles}`, :math:`N_{bonds}`),
                        dtype= :class:`numpy.uint32`
        """
        frame_0 = freud.common.convert_array(
                    frame_0, 2, dtype=np.uint32, contiguous=True,
                    dim_message="frame_0 must be a 2 dimensional array")
        if (frame_0.shape[0] != self.num_particles):
            raise ValueError(
                "the 1st dimension must match num_particles: {}".format(
                    self.num_particles))
        if (frame_0.shape[1] != self.num_bonds):
            raise ValueError(
                "the 2nd dimension must match num_bonds: {}".format(
                    self.num_bonds))
        cdef np.ndarray[uint, ndim = 2] l_frame_0 = frame_0
        with nogil:
            self.thisptr.initialize(< unsigned int*> l_frame_0.data)

    def compute(self, frame_0, frame_1):
        """
        Calculates the changes in bonding states from one frame to the next.

        :param frame_0: current/previous bonding frame (as output from
                        :py:class:`.BondingR12` modules)
        :param frame_1: next/current bonding frame (as output from
                        :py:class:`.BondingR12` modules)
        :type frame_0: :class:`numpy.ndarray`,
                        shape=(:math:`N_{particles}`, :math:`N_{bonds}`),
                        dtype= :class:`numpy.uint32`
        :type frame_1: :class:`numpy.ndarray`,
                        shape=(:math:`N_{particles}`, :math:`N_{bonds}`),
                        dtype= :class:`numpy.uint32`
        """
        frame_0 = freud.common.convert_array(
                    frame_0, 2, dtype=np.uint32, contiguous=True,
                    dim_message="frame_0 must be a 2 dimensional array")
        frame_1 = freud.common.convert_array(
            frame_1, 2, dtype=np.uint32, contiguous=True,
            dim_message="frame_1 must be a 2 dimensional array")

        cdef np.ndarray[uint, ndim = 2] l_frame_0 = frame_0
        cdef np.ndarray[uint, ndim = 2] l_frame_1 = frame_1
        with nogil:
            self.thisptr.compute(
                < unsigned int*> l_frame_0.data,
                < unsigned int*> l_frame_1.data)
        return self

    @property
    def bond_lifetimes(self):
        """The bond lifetimes
        """
        return self.getBondLifetimes()

    def getBondLifetimes(self):
        """The bond lifetimes

        :return: lifetime of bonds
        :rtype: :class:`numpy.ndarray`,
                shape=(:math:`N_{particles}`, varying),
                dtype= :class:`numpy.uint32`
        """
        bonds = self.thisptr.getBondLifetimes()
        return bonds

    @property
    def overall_lifetimes(self):
        """The overall lifetimes
        """
        return self.getOverallLifetimes()

    def getOverallLifetimes(self):
        """The overall lifetimes
        :return: lifetime of bonds
        :rtype: :class:`numpy.ndarray`,
                shape=(:math:`N_{particles}`, varying),
                dtype= :class:`numpy.uint32`
        """
        bonds = self.thisptr.getOverallLifetimes()
        ret_bonds = np.copy(np.asarray(bonds, dtype=np.uint32))
        return ret_bonds

    @property
    def transition_matrix(self):
        """The transition matrix
        """
        return self.getTransitionMatrix()

    def getTransitionMatrix(self):
        """The transition matrix
        :return: transition matrix
        :rtype: :class:`numpy.ndarray`
        """
        cdef unsigned int * trans_matrix = self.thisptr.getTransitionMatrix(
                                            ).get()
        cdef np.npy_intp nbins[2]
        nbins[0] = <np.npy_intp > self.num_bonds
        nbins[1] = <np.npy_intp > self.num_bonds
        cdef np.ndarray[np.uint32_t, ndim = 2
                        ] result = np.PyArray_SimpleNewFromData(
                        2, nbins, np.NPY_UINT32, < void*>trans_matrix)
        return result

    @property
    def num_frames(self):
        """Get number of frames calculated
        """
        return self.getNumFrames()

    def getNumFrames(self):
        """
        Get number of frames calculated

        :return: number of frames
        :rtype: unsigned int
        """
        return self.thisptr.getNumFrames()

    @property
    def num_particles(self):
        """Get number of particles being tracked
        """
        return self.getNumParticles()

    def getNumParticles(self):
        """
        Get number of particles being tracked

        :return: number of particles
        :rtype: unsigned int
        """
        return self.thisptr.getNumParticles()

    @property
    def num_bonds(self):
        """Get number of bonds being tracked
        """
        return self.getNumBonds()

    def getNumBonds(self):
        """
        Get number of bonds tracked

        :return: number of bonds
        :rtype: unsigned int
        """
        return self.thisptr.getNumBonds()

cdef class BondingR12:
    """Compute the bonds each particle in the system.

    For each particle in the system determine which other particles are in
    which bonding sites.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    :param r_max: distance to search for bonds
    :param bond_map: 3D array containing the bond index for each r, t2, t1
                    coordinate
    :param bond_list: list containing the bond indices to be tracked
                    bond_list[i] = bond_index
    :type r_max: float
    :type bond_map: :class:`numpy.ndarray`
    :type bond_list: :class:`numpy.ndarray`
    """
    cdef bond.BondingR12 * thisptr
    cdef rmax

    def __cinit__(self, float r_max, bond_map, bond_list):
        # extract nr, nt from the bond_map
        n_r = bond_map.shape[0]
        n_t2 = bond_map.shape[1]
        n_t1 = bond_map.shape[2]
        n_bonds = bond_list.shape[0]
        cdef np.ndarray[uint, ndim = 3] l_bond_map = bond_map
        cdef np.ndarray[uint, ndim = 1] l_bond_list = bond_list
        self.thisptr = new bond.BondingR12(
                r_max, n_r, n_t2, n_t1, n_bonds,
                < unsigned int*>l_bond_map.data,
                < unsigned int*>l_bond_list.data)
        self.rmax = r_max

    def __dealloc__(self):
        del self.thisptr

    def compute(self, box, ref_points, ref_orientations, points, orientations,
                nlist=None):
        """
        Calculates the correlation function and adds to the current histogram.

        :param box: simulation box
        :param ref_points: points to calculate the bonding
        :param ref_orientations: orientations as angles to use in computation
        :param points: points to calculate the bonding
        :param orientations: orientations as angles to use in computation
        :param nlist: :py:class:`freud.locality.NeighborList` object to use to
                        find bonds
        :type box: :py:meth:`freud.box.Box`
        :type ref_points: :class:`numpy.ndarray`,
                            shape=(:math:`N_{particles}`, 3),
                            dtype= :class:`numpy.float32`
        :type ref_orientations: :class:`numpy.ndarray`,
                            shape=(:math:`N_{particles}`),
                            dtype= :class:`numpy.float32`
        :type points: :class:`numpy.ndarray`,
                            shape=(:math:`N_{particles}`, 3),
                            dtype= :class:`numpy.float32`
        :type orientations: :class:`numpy.ndarray`,
                            shape=(:math:`N_{particles}`),
                            dtype= :class:`numpy.float32`
        :type nlist: :py:class:`freud.locality.NeighborList`
        """
        ref_points = freud.common.convert_array(
                ref_points, 2, dtype=np.float32, contiguous=True,
                dim_message="ref_points must be a 2 dimensional array")
        if ref_points.shape[1] != 3:
            raise TypeError('ref_points should be an Nx3 array')

        ref_orientations = freud.common.convert_array(
                ref_orientations, 1, dtype=np.float32, contiguous=True,
                dim_message="ref_orientations must be a 1 dimensional array")

        points = freud.common.convert_array(
                points, 2, dtype=np.float32, contiguous=True,
                dim_message="points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        orientations = freud.common.convert_array(
                orientations, 1, dtype=np.float32, contiguous=True,
                dim_message="orientations must be a 1 dimensional array")

        defaulted_nlist = make_default_nlist(
            box, ref_points, points, self.rmax, nlist, None)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        cdef locality.NeighborList * nlist_ptr = nlist_.get_ptr()

        cdef np.ndarray[float, ndim = 2] l_ref_points = ref_points
        cdef np.ndarray[float, ndim = 1] l_ref_orientations = ref_orientations
        cdef np.ndarray[float, ndim = 2] l_points = points
        cdef np.ndarray[float, ndim = 1] l_orientations = orientations
        cdef unsigned int n_ref = <unsigned int > ref_points.shape[0]
        cdef unsigned int n_p = <unsigned int > points.shape[0]
        cdef _box.Box l_box = _box.Box(
                box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
                box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.compute(
                    l_box, nlist_ptr,
                    < vec3[float]*> l_ref_points.data,
                    < float*> l_ref_orientations.data, n_ref,
                    < vec3[float]*> l_points.data,
                    < float*> l_orientations.data, n_p)
        return self

    @property
    def bonds(self):
        """The particle bonds
        """
        return self.getBonds()

    def getBonds(self):
        """Return the particle bonds

        :return: particle bonds
        :rtype: :class:`numpy.ndarray`
        """
        cdef unsigned int * bonds = self.thisptr.getBonds().get()
        cdef np.npy_intp nbins[2]
        nbins[0] = <np.npy_intp > self.thisptr.getNumParticles()
        nbins[1] = <np.npy_intp > self.thisptr.getNumBonds()
        cdef np.ndarray[np.uint32_t, ndim = 2
                        ] result = np.PyArray_SimpleNewFromData(
                        2, nbins, np.NPY_UINT32, < void*>bonds)
        return result

    @property
    def box(self):
        """Get the box used in the calculation
        """
        return self.getBox()

    def getBox(self):
        """
        Get the box used in the calculation

        :return: freud Box
        :rtype: :py:meth:`freud.box.Box()`
        """
        return BoxFromCPP(< box.Box > self.thisptr.getBox())

    @property
    def list_map(self):
        """Get the dict used to map list idx to bond idx
        """
        return self.getListMap()

    def getListMap(self):
        """
        Get the dict used to map list idx to bond idx

        :return: list_map
        :rtype: dict

        >>> list_idx = list_map[bond_idx]
        """
        return self.thisptr.getListMap()

    @property
    def rev_list_map(self):
        """Get the dict used to map list idx to bond idx
        """
        return self.getRevListMap()

    def getRevListMap(self):
        """
        Get the dict used to map list idx to bond idx

        :return: list_map
        :rtype: dict

        >>> bond_idx = list_map[list_idx]
        """
        return self.thisptr.getRevListMap()

cdef class BondingXY2D:
    """Compute the bonds each particle in the system.

    For each particle in the system determine which other particles are in
    which bonding sites.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    :param x_max: maximum x distance at which to search for bonds
    :param y_max: maximum y distance at which to search for bonds
    :param bond_map: 3D array containing the bond index for each x, y
                        coordinate
    :param bond_list: list containing the bond indices to be tracked
                        bond_list[i] = bond_index
    :type x_max: float
    :type y_max: float
    :type bond_map: :class:`numpy.ndarray`
    :type bond_list: :class:`numpy.ndarray`
    """
    cdef bond.BondingXY2D * thisptr
    cdef rmax

    def __cinit__(self, float x_max, float y_max, bond_map, bond_list):
        # extract nr, nt from the bond_map
        n_y = bond_map.shape[0]
        n_x = bond_map.shape[1]
        n_bonds = bond_list.shape[0]
        bond_map = np.require(bond_map, requirements=["C"])
        bond_list = np.require(bond_list, requirements=["C"])
        cdef np.ndarray[uint, ndim = 2] l_bond_map = bond_map
        cdef np.ndarray[uint, ndim = 1] l_bond_list = bond_list
        self.thisptr = new bond.BondingXY2D(
                x_max, y_max, n_x, n_y, n_bonds,
                < unsigned int*>l_bond_map.data,
                < unsigned int*>l_bond_list.data)
        self.rmax = np.sqrt(x_max**2 + y_max**2)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, box, ref_points, ref_orientations, points, orientations,
                nlist=None):
        """ Calculates the correlation function and adds to the current
        histogram.

        :param box: simulation box
        :param ref_points: points to calculate the bonding
        :param ref_orientations: orientations as angles to use in computation
        :param points: points to calculate the bonding
        :param orientations: orientations as angles to use in computation
        :param nlist: :py:class:`freud.locality.NeighborList` object to use to
                       find bonds
        :type box: :py:meth:`freud.box.Box`
        :type ref_points: :class:`numpy.ndarray`,
                            shape=(:math:`N_{particles}`, 3),
                            dtype= :class:`numpy.float32`
        :type ref_orientations: :class:`numpy.ndarray`,
                                shape=(:math:`N_{particles}`),
                                dtype= :class:`numpy.float32`
        :type points: :class:`numpy.ndarray`,
                        shape=(:math:`N_{particles}`, 3),
                        dtype= :class:`numpy.float32`
        :type orientations: :class:`numpy.ndarray`,
                        shape=(:math:`N_{particles}`),
                        dtype= :class:`numpy.float32`
        :type nlist: :py:class:`freud.locality.NeighborList`
        """
        ref_points = freud.common.convert_array(
                ref_points, 2, dtype=np.float32, contiguous=True,
                dim_message="ref_points must be a 2 dimensional array")
        if ref_points.shape[1] != 3:
            raise TypeError('ref_points should be an Nx3 array')

        ref_orientations = freud.common.convert_array(
                ref_orientations, 1, dtype=np.float32, contiguous=True,
                dim_message="ref_orientations must be a 1 dimensional array")

        points = freud.common.convert_array(
                points, 2, dtype=np.float32, contiguous=True,
                dim_message="points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        orientations = freud.common.convert_array(
                orientations, 1, dtype=np.float32, contiguous=True,
                dim_message="orientations must be a 1 dimensional array")

        defaulted_nlist = make_default_nlist(
            box, ref_points, points, self.rmax, nlist, None)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        cdef locality.NeighborList * nlist_ptr = nlist_.get_ptr()

        cdef np.ndarray[float, ndim = 2] l_ref_points = ref_points
        cdef np.ndarray[float, ndim = 1] l_ref_orientations = ref_orientations
        cdef np.ndarray[float, ndim = 2] l_points = points
        cdef np.ndarray[float, ndim = 1] l_orientations = orientations
        cdef unsigned int n_ref = <unsigned int > ref_points.shape[0]
        cdef unsigned int n_p = <unsigned int > points.shape[0]
        cdef _box.Box l_box = _box.Box(
                box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
                box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.compute(
                    l_box, nlist_ptr,
                    < vec3[float]*> l_ref_points.data,
                    < float*> l_ref_orientations.data,
                    n_ref,
                    < vec3[float]*> l_points.data,
                    < float*> l_orientations.data, n_p)
        return self

    @property
    def bonds(self):
        """The particle bonds
        """
        return self.getBonds()

    def getBonds(self):
        """Return the particle bonds

        :return: particle bonds
        :rtype: :class:`numpy.ndarray`
        """
        cdef unsigned int * bonds = self.thisptr.getBonds().get()
        cdef np.npy_intp nbins[2]
        nbins[0] = <np.npy_intp > self.thisptr.getNumParticles()
        nbins[1] = <np.npy_intp > self.thisptr.getNumBonds()
        cdef np.ndarray[np.uint32_t, ndim = 2
                        ] result = np.PyArray_SimpleNewFromData(
                            2, nbins, np.NPY_UINT32, < void*>bonds)
        return result

    @property
    def box(self):
        """Get the box used in the calculation
        """
        return self.getBox()

    def getBox(self):
        """
        Get the box used in the calculation

        :return: freud Box
        :rtype: :py:meth:`freud.box.Box()`
        """
        return BoxFromCPP(< box.Box > self.thisptr.getBox())

    @property
    def list_map(self):
        """Get the dict used to map list idx to bond idx
        """
        return self.getListMap()

    def getListMap(self):
        """
        Get the dict used to map list idx to bond idx

        :return: list_map
        :rtype: dict

        >>> list_idx = list_map[bond_idx]
        """
        return self.thisptr.getListMap()

    @property
    def rev_list_map(self):
        """Get the dict used to map list idx to bond idx
        """
        return self.getRevListMap()

    def getRevListMap(self):
        """
        Get the dict used to map list idx to bond idx

        :return: list_map
        :rtype: dict

        >>> bond_idx = list_map[list_idx]
        """
        return self.thisptr.getRevListMap()

cdef class BondingXYT:
    """Compute the bonds each particle in the system.

    For each particle in the system determine which other particles are in
    which bonding sites.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    :param x_max: maximum x distance at which to search for bonds
    :param y_max: maximum y distance at which to search for bonds
    :param bond_map: 3D array containing the bond index for each x, y
                    coordinate
    :param bond_list: list containing the bond indices to be tracked
                    bond_list[i] = bond_index
    :type x_max: float
    :type y_max: float
    :type bond_map: :class:`numpy.ndarray`
    :type bond_list: :class:`numpy.ndarray`
    """
    cdef bond.BondingXYT * thisptr
    cdef rmax

    def __cinit__(self, float x_max, float y_max, bond_map, bond_list):
        # extract nr, nt from the bond_map
        n_t = bond_map.shape[0]
        n_y = bond_map.shape[1]
        n_x = bond_map.shape[2]
        n_bonds = bond_list.shape[0]
        bond_map = np.require(bond_map, requirements=["C"])
        bond_list = np.require(bond_list, requirements=["C"])
        cdef np.ndarray[uint, ndim = 3] l_bond_map = bond_map
        cdef np.ndarray[uint, ndim = 1] l_bond_list = bond_list
        self.thisptr = new bond.BondingXYT(
                x_max, y_max, n_x, n_y, n_t, n_bonds,
                < unsigned int*>l_bond_map.data,
                < unsigned int*>l_bond_list.data)
        self.rmax = np.sqrt(x_max**2 + y_max**2)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, box, ref_points, ref_orientations, points, orientations,
                nlist=None):
        """
        Calculates the correlation function and adds to the current histogram.

        :param box: simulation box
        :param ref_points: points to calculate the bonding
        :param ref_orientations: orientations as angles to use in computation
        :param points: points to calculate the bonding
        :param orientations: orientations as angles to use in computation
        :param nlist: :py:class:`freud.locality.NeighborList` object to use to
                        find bonds
        :type box: :py:meth:`freud.box.Box`
        :type ref_points: :class:`numpy.ndarray`,
                            shape=(:math:`N_{particles}`, 3),
                            dtype= :class:`numpy.float32`
        :type ref_orientations: :class:`numpy.ndarray`,
                                shape=(:math:`N_{particles}`),
                                dtype= :class:`numpy.float32`
        :type points: :class:`numpy.ndarray`,
                        shape=(:math:`N_{particles}`, 3),
                        dtype= :class:`numpy.float32`
        :type orientations: :class:`numpy.ndarray`,
                            shape=(:math:`N_{particles}`),
                            dtype= :class:`numpy.float32`
        :type nlist: :py:class:`freud.locality.NeighborList`
        """
        ref_points = freud.common.convert_array(
                ref_points, 2, dtype=np.float32, contiguous=True,
                dim_message="ref_points must be a 2 dimensional array")
        if ref_points.shape[1] != 3:
            raise TypeError('ref_points should be an Nx3 array')

        ref_orientations = freud.common.convert_array(
                ref_orientations, 1, dtype=np.float32, contiguous=True,
                dim_message="ref_orientations must be a 1 dimensional array")

        points = freud.common.convert_array(
                points, 2, dtype=np.float32, contiguous=True,
                dim_message="points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        orientations = freud.common.convert_array(
                orientations, 1, dtype=np.float32, contiguous=True,
                dim_message="orientations must be a 1 dimensional array")

        defaulted_nlist = make_default_nlist(
            box, ref_points, points, self.rmax, nlist, None)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        cdef locality.NeighborList * nlist_ptr = nlist_.get_ptr()

        cdef np.ndarray[float, ndim = 2] l_ref_points = ref_points
        cdef np.ndarray[float, ndim = 1] l_ref_orientations = ref_orientations
        cdef np.ndarray[float, ndim = 2] l_points = points
        cdef np.ndarray[float, ndim = 1] l_orientations = orientations
        cdef unsigned int n_ref = <unsigned int > ref_points.shape[0]
        cdef unsigned int n_p = <unsigned int > points.shape[0]
        cdef _box.Box l_box = _box.Box(
                box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
                box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.compute(
                    l_box, nlist_ptr,
                    < vec3[float]*> l_ref_points.data,
                    < float*> l_ref_orientations.data, n_ref,
                    < vec3[float]*> l_points.data,
                    < float*> l_orientations.data, n_p)
        return self

    @property
    def bonds(self):
        """The particle bonds
        """
        return self.getBonds()

    def getBonds(self):
        """Return the particle bonds

        :return: particle bonds
        :rtype: :class:`numpy.ndarray`
        """
        cdef unsigned int * bonds = self.thisptr.getBonds().get()
        cdef np.npy_intp nbins[2]
        nbins[0] = <np.npy_intp > self.thisptr.getNumParticles()
        nbins[1] = <np.npy_intp > self.thisptr.getNumBonds()
        cdef np.ndarray[np.uint32_t, ndim = 2
                        ] result = np.PyArray_SimpleNewFromData(
                        2, nbins, np.NPY_UINT32, < void*>bonds)
        return result

    @property
    def box(self):
        """Get the box used in the calculation
        """
        return self.getBox()

    def getBox(self):
        """
        Get the box used in the calculation

        :return: freud Box
        :rtype: :py:meth:`freud.box.Box()`
        """
        return BoxFromCPP(< box.Box > self.thisptr.getBox())

    @property
    def list_map(self):
        """Get the dict used to map list idx to bond idx
        """
        return self.getListMap()

    def getListMap(self):
        """
        Get the dict used to map list idx to bond idx

        :return: list_map
        :rtype: dict

        >>> list_idx = list_map[bond_idx]
        """
        return self.thisptr.getListMap()

    @property
    def rev_list_map(self):
        """Get the dict used to map list idx to bond idx
        """
        return self.getRevListMap()

    def getRevListMap(self):
        """
        Get the dict used to map list idx to bond idx

        :return: list_map
        :rtype: dict

        >>> bond_idx = list_map[list_idx]
        """
        return self.thisptr.getRevListMap()

cdef class BondingXYZ:
    """Compute the bonds each particle in the system.

    For each particle in the system determine which other particles are in
    which bonding sites.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    :param x_max: maximum x distance at which to search for bonds
    :param y_max: maximum y distance at which to search for bonds
    :param z_max: maximum z distance at which to search for bonds
    :param bond_map: 3D array containing the bond index for each x, y, z
                    coordinate
    :param bond_list: list containing the bond indices to be tracked
                    bond_list[i] = bond_index
    :type x_max: float
    :type y_max: float
    :type z_max: float
    :type bond_map: :class:`numpy.ndarray`
    :type bond_list: :class:`numpy.ndarray`
    """
    cdef bond.BondingXYZ * thisptr
    cdef rmax

    def __cinit__(self, float x_max, float y_max, float z_max, bond_map,
            bond_list):
        # extract nr, nt from the bond_map
        n_z = bond_map.shape[0]
        n_y = bond_map.shape[1]
        n_x = bond_map.shape[2]
        n_bonds = bond_list.shape[0]
        bond_map = np.require(bond_map, requirements=["C"])
        bond_list = np.require(bond_list, requirements=["C"])
        cdef np.ndarray[uint, ndim = 3] l_bond_map = bond_map
        cdef np.ndarray[uint, ndim = 1] l_bond_list = bond_list
        self.thisptr = new bond.BondingXYZ(
                x_max, y_max, z_max, n_x, n_y, n_z, n_bonds,
                < unsigned int*>l_bond_map.data,
                < unsigned int*>l_bond_list.data)
        self.rmax = np.sqrt(x_max**2 + y_max**2 + z_max**2)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, box, ref_points, ref_orientations, points, orientations,
                nlist=None):
        """
        Calculates the correlation function and adds to the current histogram.

        :param box: simulation box
        :param ref_points: points to calculate the bonding
        :param ref_orientations: orientations as quaternions to use in
                                computation
        :param points: points to calculate the bonding
        :param orientations: orientations as quaternions to use in computation
        :param nlist: :py:class:`freud.locality.NeighborList` object to use to
                        find bonds
        :type box: :py:meth:`freud.box.Box`
        :type ref_points: :class:`numpy.ndarray`,
                            shape=(:math:`N_{particles}`, 3),
                            dtype= :class:`numpy.float32`
        :type ref_orientations: :class:`numpy.ndarray`,
                                    shape=(:math:`N_{particles}`, 4),
                                    dtype= :class:`numpy.float32`
        :type points: :class:`numpy.ndarray`,
                        shape=(:math:`N_{particles}`, 3),
                        dtype= :class:`numpy.float32`
        :type orientations: :class:`numpy.ndarray`,
                            shape=(:math:`N_{particles}`, 4),
                            dtype= :class:`numpy.float32`
        :type nlist: :py:class:`freud.locality.NeighborList`
        """
        ref_points = freud.common.convert_array(
                ref_points, 2, dtype=np.float32, contiguous=True,
                dim_message="ref_points must be a 2 dimensional array")
        if ref_points.shape[1] != 3:
            raise TypeError('ref_points should be an Nx3 array')

        ref_orientations = freud.common.convert_array(
                ref_orientations, 2, dtype=np.float32, contiguous=True,
                dim_message="ref_orientations must be a 2 dimensional array")
        if ref_orientations.shape[1] != 4:
            raise ValueError(
                "the 2nd dimension must have 4 values: q0, q1, q2, q3")

        points = freud.common.convert_array(
                points, 2, dtype=np.float32, contiguous=True,
                dim_message="points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        orientations = freud.common.convert_array(
                orientations, 2, dtype=np.float32, contiguous=True,
                dim_message="orientations must be a 2 dimensional array")
        if orientations.shape[1] != 4:
            raise ValueError(
                "the 2nd dimension must have 4 values: q0, q1, q2, q3")

        defaulted_nlist = make_default_nlist(
            box, ref_points, points, self.rmax, nlist, None)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        cdef locality.NeighborList * nlist_ptr = nlist_.get_ptr()

        cdef np.ndarray[float, ndim = 2] l_ref_points = ref_points
        cdef np.ndarray[float, ndim = 2] l_ref_orientations = ref_orientations
        cdef np.ndarray[float, ndim = 2] l_points = points
        cdef np.ndarray[float, ndim = 2] l_orientations = orientations
        cdef unsigned int n_ref = <unsigned int > ref_points.shape[0]
        cdef unsigned int n_p = <unsigned int > points.shape[0]
        cdef _box.Box l_box = _box.Box(
                box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
                box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.compute(
                    l_box, nlist_ptr,
                    < vec3[float]*> l_ref_points.data,
                    < quat[float]*> l_ref_orientations.data,
                    n_ref,
                    < vec3[float]*> l_points.data,
                    < quat[float]*> l_orientations.data,
                    n_p)
        return self

    @property
    def bonds(self):
        """The particle bonds
        """
        return self.getBonds()

    def getBonds(self):
        """Return the particle bonds

        :return: particle bonds
        :rtype: :class:`numpy.ndarray`
        """
        cdef unsigned int * bonds = self.thisptr.getBonds().get()
        cdef np.npy_intp nbins[2]
        nbins[0] = <np.npy_intp > self.thisptr.getNumParticles()
        nbins[1] = <np.npy_intp > self.thisptr.getNumBonds()
        cdef np.ndarray[np.uint32_t, ndim = 2
                        ] result = np.PyArray_SimpleNewFromData(
                        2, nbins, np.NPY_UINT32, < void*>bonds)
        return result

    @property
    def box(self):
        """Get the box used in the calculation
        """
        return self.getBox()

    def getBox(self):
        """
        Get the box used in the calculation

        :return: freud Box
        :rtype: :py:meth:`freud.box.Box()`
        """
        return BoxFromCPP(< box.Box > self.thisptr.getBox())

    @property
    def list_map(self):
        """Get the dict used to map list idx to bond idx
        """
        return self.getListMap()

    def getListMap(self):
        """
        Get the dict used to map list idx to bond idx

        :return: list_map
        :rtype: dict

        >>> list_idx = list_map[bond_idx]
        """
        return self.thisptr.getListMap()

    @property
    def rev_list_map(self):
        """Get the dict used to map list idx to bond idx
        """
        return self.getRevListMap()

    def getRevListMap(self):
        """
        Get the dict used to map list idx to bond idx

        :return: list_map
        :rtype: dict

        >>> bond_idx = list_map[list_idx]
        """
        return self.thisptr.getRevListMap()
