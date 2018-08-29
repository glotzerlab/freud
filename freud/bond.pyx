# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

R"""
The bond module allows for the computation of bonds as defined by a map.
Depending on the coordinate system desired, either a two or three dimensional
array is supplied, with each element containing the bond index mapped to the
pair geometry of that element. The user provides a list of indices to track, so
that not all bond indices contained in the bond map need to be tracked in
computation.

The bond module is designed to take in arrays using the same coordinate systems
as the :doc:`pmft` in freud.

.. note::
    The coordinate system in which the calculation is performed is not the same
    as the coordinate system in which particle positions and orientations
    should be supplied. Only certain coordinate systems are available for
    certain particle positions and orientations:

    * 2D particle coordinates (position: [:math:`x`, :math:`y`, :math:`0`],
      orientation: :math:`\theta`):

        * :math:`r`, :math:`\theta_1`, :math:`\theta_2`.
        * :math:`x`, :math:`y`.
        * :math:`x`, :math:`y`, :math:`\theta`.

    * 3D particle coordinates:

        * :math:`x`, :math:`y`, :math:`z`.
"""

import numpy as np
import warnings
import freud.common
import freud.locality
from freud.errors import FreudDeprecationWarning

from cython.operator cimport dereference
from freud.util._VectorMath cimport vec3, quat
from libcpp.map cimport map

cimport freud.locality
cimport freud._bond
cimport numpy as np
cimport freud.box


# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef class BondingAnalysis:
    """Analyze the bond lifetimes and flux present in the system.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    Args:
        num_particles (unsigned int):
            Number of particles over which to calculate bonds.
        num_bonds (unsigned int):
            Number of bonds to track.

    Attributes:
        bond_lifetimes ((:math:`N_{particles}`, varying) \
:class:`numpy.ndarray`):
            Bond lifetimes.
        overall_lifetimes ((:math:`N_{particles}`, varying) \
:class:`numpy.ndarray`):
            Overall bond lifetimes.
        transition_matrix (:class:`numpy.ndarray`):
            Transition matrix.
        num_frames (unsigned int):
            Number of frames calculated.
        num_particles (unsigned int):
            Number of tracked particles.
        num_bonds (unsigned int):
            Number of tracked bonds.
    """
    cdef freud._bond.BondingAnalysis * thisptr
    cdef unsigned int num_particles
    cdef unsigned int num_bonds

    def __cinit__(self, int num_particles, int num_bonds):
        self.num_particles = num_particles
        self.num_bonds = num_bonds
        self.thisptr = new freud._bond.BondingAnalysis(num_particles,
                                                       num_bonds)

    def __dealloc__(self):
        del self.thisptr

    def initialize(self, frame_0):
        """Calculates the changes in bonding states from one frame to the next.

        Args:
            frame_0 ((:math:`N_{particles}`, :math:`N_{bonds}`) \
:class:`numpy.ndarray`):
                First bonding frame (as output from :py:class:`~.BondingR12`
                modules).
        """
        frame_0 = freud.common.convert_array(
            frame_0, 2, dtype=np.uint32, contiguous=True, array_name="frame_0")
        if (frame_0.shape[0] != self.num_particles):
            raise ValueError(
                "The 1st dimension must match num_particles: {}".format(
                    self.num_particles))
        if (frame_0.shape[1] != self.num_bonds):
            raise ValueError(
                "The 2nd dimension must match num_bonds: {}".format(
                    self.num_bonds))
        cdef np.ndarray[uint, ndim=2] l_frame_0 = frame_0
        with nogil:
            self.thisptr.initialize(<unsigned int*> l_frame_0.data)

    def compute(self, frame_0, frame_1):
        """Calculates the changes in bonding states from one frame to the next.

        Args:
            frame_0 ((:math:`N_{particles}`, :math:`N_{bonds}`) \
:class:`numpy.ndarray`):
                Current/previous bonding frame (as output from
                :py:class:`.BondingR12` modules).
            frame_1 ((:math:`N_{particles}`, :math:`N_{bonds}`) \
:class:`numpy.ndarray`):
                Next/current bonding frame (as output from
                :py:class:`.BondingR12` modules).
        """
        frame_0 = freud.common.convert_array(
            frame_0, 2, dtype=np.uint32, contiguous=True, array_name="frame_0")
        frame_1 = freud.common.convert_array(
            frame_1, 2, dtype=np.uint32, contiguous=True, array_name="frame_1")

        cdef np.ndarray[uint, ndim=2] l_frame_0 = frame_0
        cdef np.ndarray[uint, ndim=2] l_frame_1 = frame_1
        with nogil:
            self.thisptr.compute(
                <unsigned int*> l_frame_0.data,
                <unsigned int*> l_frame_1.data)
        return self

    @property
    def bond_lifetimes(self):
        bonds = self.thisptr.getBondLifetimes()
        return bonds

    def getBondLifetimes(self):
        warnings.warn("The getBondLifetimes function is deprecated in favor "
                      "of the bond_lifetimes class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.bond_lifetimes

    @property
    def overall_lifetimes(self):
        bonds = self.thisptr.getOverallLifetimes()
        ret_bonds = np.copy(np.asarray(bonds, dtype=np.uint32))
        return ret_bonds

    def getOverallLifetimes(self):
        warnings.warn("The getOverallLifetimes function is deprecated in "
                      "favor of the overall_lifetimes class attribute and "
                      "will be removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.overall_lifetimes

    @property
    def transition_matrix(self):
        cdef unsigned int * trans_matrix = \
            self.thisptr.getTransitionMatrix().get()
        cdef np.npy_intp nbins[2]
        nbins[0] = <np.npy_intp> self.num_bonds
        nbins[1] = <np.npy_intp> self.num_bonds
        cdef np.ndarray[np.uint32_t, ndim=2] result = \
            np.PyArray_SimpleNewFromData(
                2, nbins, np.NPY_UINT32, <void*> trans_matrix)
        return result

    def getTransitionMatrix(self):
        warnings.warn("The getTransitionMatrix function is deprecated in "
                      "favor of the transition_matrix class attribute and "
                      "will be removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.transition_matrix

    @property
    def num_frames(self):
        return self.thisptr.getNumFrames()

    def getNumFrames(self):
        warnings.warn("The getNumFrames function is deprecated in favor "
                      "of the num_frames class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.num_frames

    @property
    def num_particles(self):
        return self.thisptr.getNumParticles()

    def getNumParticles(self):
        warnings.warn("The getNumParticles function is deprecated in favor "
                      "of the num_particles class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.num_particles

    @property
    def num_bonds(self):
        return self.thisptr.getNumBonds()

    def getNumBonds(self):
        warnings.warn("The getNumBonds function is deprecated in favor "
                      "of the num_bonds class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.num_bonds

cdef class BondingR12:
    """Compute bonds in a 2D system using a (:math:`r`, :math:`\\theta_1`,
    :math:`\\theta_2`) coordinate system.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    Args:
        r_max (float):
            Distance to search for bonds.
        bond_map (:class:`numpy.ndarray`):
            3D array containing the bond index for each r, :math:`\\theta_2`,
            :math:`\\theta_1` coordinate.
        bond_list (:class:`numpy.ndarray`):
            List containing the bond indices to be tracked,
            :code:`bond_list[i] = bond_index`.

    Attributes:
        bonds (:class:`numpy.ndarray`):
            Particle bonds.
        box (:py:class:`freud.box.Box`):
            Box used in the calculation.
        list_map (dict):
            The dict used to map bond index to list index.
        rev_list_map (dict):
            The dict used to map list idx to bond idx.
    """
    cdef freud._bond.BondingR12 * thisptr
    cdef rmax

    def __cinit__(self, float r_max, bond_map, bond_list):
        # extract nr, nt from the bond_map
        n_r = bond_map.shape[0]
        n_t2 = bond_map.shape[1]
        n_t1 = bond_map.shape[2]
        n_bonds = bond_list.shape[0]
        cdef np.ndarray[uint, ndim=3] l_bond_map = bond_map
        cdef np.ndarray[uint, ndim=1] l_bond_list = bond_list
        self.thisptr = new freud._bond.BondingR12(
            r_max, n_r, n_t2, n_t1, n_bonds,
            <unsigned int*> l_bond_map.data,
            <unsigned int*> l_bond_list.data)
        self.rmax = r_max

    def __dealloc__(self):
        del self.thisptr

    def compute(self, box, ref_points, ref_orientations, points=None,
                orientations=None, nlist=None):
        """Calculates the bonds.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points used to calculate the bonding.
            ref_orientations ((:math:`N_{particles}`, 1) or
            (:math:`N_{particles}`,) :class:`numpy.ndarray`):
                Reference orientations as angles to use in computation.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points used to calculate the bonding. Uses :code:`ref_points`
                if not provided or :code:`None`.
            orientations ((:math:`N_{particles}`, 1) or
            (:math:`N_{particles}`,) :class:`numpy.ndarray`, optional):
                Orientations as angles to use in computation. Uses
                :code:`ref_orientations` if not provided or :code:`None`.
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList to use to find bonds (Default value =
                :code:`None`).
        """
        cdef freud.box.Box b = freud.common.convert_box(box)

        if points is None:
            points = ref_points
        if orientations is None:
            orientations = ref_orientations

        ref_points = freud.common.convert_array(
            ref_points, 2, dtype=np.float32, contiguous=True,
            array_name="ref_points")
        if ref_points.shape[1] != 3:
            raise TypeError('ref_points should be an Nx3 array')

        ref_orientations = freud.common.convert_array(
            ref_orientations.squeeze(), 1, dtype=np.float32, contiguous=True,
            array_name="ref_orientations")

        points = freud.common.convert_array(
            points, 2, dtype=np.float32, contiguous=True, array_name="points")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        orientations = freud.common.convert_array(
            orientations.squeeze(), 1, dtype=np.float32, contiguous=True,
            array_name="orientations")

        defaulted_nlist = freud.locality.make_default_nlist(
            b, ref_points, points, self.rmax, nlist, None)
        cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]

        cdef np.ndarray[float, ndim=2] l_ref_points = ref_points
        cdef np.ndarray[float, ndim=1] l_ref_orientations = ref_orientations
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef np.ndarray[float, ndim=1] l_orientations = orientations
        cdef unsigned int n_ref = <unsigned int> ref_points.shape[0]
        cdef unsigned int n_p = <unsigned int> points.shape[0]
        with nogil:
            self.thisptr.compute(
                dereference(b.thisptr), nlist_.get_ptr(),
                <vec3[float]*> l_ref_points.data,
                <float*> l_ref_orientations.data, n_ref,
                <vec3[float]*> l_points.data,
                <float*> l_orientations.data, n_p)
        return self

    @property
    def bonds(self):
        cdef unsigned int * bonds = self.thisptr.getBonds().get()
        cdef np.npy_intp nbins[2]
        nbins[0] = <np.npy_intp> self.thisptr.getNumParticles()
        nbins[1] = <np.npy_intp> self.thisptr.getNumBonds()
        cdef np.ndarray[np.uint32_t, ndim=2] result = \
            np.PyArray_SimpleNewFromData(
                2, nbins, np.NPY_UINT32, <void*> bonds)
        return result

    def getBonds(self):
        warnings.warn("The getBonds function is deprecated in favor "
                      "of the bonds class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.bonds

    @property
    def box(self):
        return freud.box.BoxFromCPP(self.thisptr.getBox())

    def getBox(self):
        warnings.warn("The getBox function is deprecated in favor "
                      "of the box class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.box

    @property
    def list_map(self):
        return self.thisptr.getListMap()

    def getListMap(self):
        warnings.warn("The getListMap function is deprecated in favor "
                      "of the list_map class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.list_map

    @property
    def rev_list_map(self):
        return self.thisptr.getRevListMap()

    def getRevListMap(self):
        warnings.warn("The getRevListMap function is deprecated in favor "
                      "of the rev_list_map class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.rev_list_map

cdef class BondingXY2D:
    """Compute bonds in a 2D system using a (:math:`x`, :math:`y`) coordinate
    system.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    Args:
        x_max (float):
            Maximum :math:`x` distance at which to search for bonds.
        y_max (float):
            Maximum :math:`y` distance at which to search for bonds.
        bond_map (:class:`numpy.ndarray`):
            3D array containing the bond index for each :math:`x`, :math:`y`
            coordinate.
        bond_list (:class:`numpy.ndarray`):
            List containing the bond indices to be tracked,
            :code:`bond_list[i] = bond_index`.

    Attributes:
        bonds (:class:`numpy.ndarray`):
            Particle bonds.
        box (:py:class:`freud.box.Box`):
            Box used in the calculation.
        list_map (dict):
            The dict used to map bond index to list index.
        rev_list_map (dict):
            The dict used to map list idx to bond idx.
    """
    cdef freud._bond.BondingXY2D * thisptr
    cdef rmax

    def __cinit__(self, float x_max, float y_max, bond_map, bond_list):
        # extract nr, nt from the bond_map
        n_y = bond_map.shape[0]
        n_x = bond_map.shape[1]
        n_bonds = bond_list.shape[0]
        bond_map = np.require(bond_map, requirements=["C"])
        bond_list = np.require(bond_list, requirements=["C"])
        cdef np.ndarray[uint, ndim=2] l_bond_map = bond_map
        cdef np.ndarray[uint, ndim=1] l_bond_list = bond_list
        self.thisptr = new freud._bond.BondingXY2D(
            x_max, y_max, n_x, n_y, n_bonds,
            <unsigned int*> l_bond_map.data,
            <unsigned int*> l_bond_list.data)
        self.rmax = np.sqrt(x_max**2 + y_max**2)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, box, ref_points, ref_orientations, points=None,
                orientations=None, nlist=None):
        """Calculates the bonds.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points used to calculate the bonding.
            ref_orientations ((:math:`N_{particles}`, 1) or
            (:math:`N_{particles}`,) :class:`numpy.ndarray`):
                Reference orientations as angles to use in computation.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points used to calculate the bonding. Uses :code:`ref_points`
                if not provided or :code:`None`.
            orientations ((:math:`N_{particles}`, 1) or
            (:math:`N_{particles}`,) :class:`numpy.ndarray`, optional):
                Orientations as angles to use in computation. Uses
                :code:`ref_orientations` if not provided or :code:`None`.
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList to use to find bonds (Default value =
                :code:`None`).
        """
        cdef freud.box.Box b = freud.common.convert_box(box)

        if points is None:
            points = ref_points
        if orientations is None:
            orientations = ref_orientations

        ref_points = freud.common.convert_array(
            ref_points, 2, dtype=np.float32, contiguous=True,
            array_name="ref_points")
        if ref_points.shape[1] != 3:
            raise TypeError('ref_points should be an Nx3 array')

        ref_orientations = freud.common.convert_array(
            ref_orientations.squeeze(), 1, dtype=np.float32, contiguous=True,
            array_name="ref_orientations")

        points = freud.common.convert_array(
            points, 2, dtype=np.float32, contiguous=True, array_name="points")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        orientations = freud.common.convert_array(
            orientations.squeeze(), 1, dtype=np.float32, contiguous=True,
            array_name="orientations")

        defaulted_nlist = freud.locality.make_default_nlist(
            b, ref_points, points, self.rmax, nlist, None)
        cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]

        cdef np.ndarray[float, ndim=2] l_ref_points = ref_points
        cdef np.ndarray[float, ndim=1] l_ref_orientations = ref_orientations
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef np.ndarray[float, ndim=1] l_orientations = orientations
        cdef unsigned int n_ref = <unsigned int> ref_points.shape[0]
        cdef unsigned int n_p = <unsigned int> points.shape[0]
        with nogil:
            self.thisptr.compute(
                dereference(b.thisptr), nlist_.get_ptr(),
                <vec3[float]*> l_ref_points.data,
                <float*> l_ref_orientations.data,
                n_ref,
                <vec3[float]*> l_points.data,
                <float*> l_orientations.data, n_p)
        return self

    @property
    def bonds(self):
        cdef unsigned int * bonds = self.thisptr.getBonds().get()
        cdef np.npy_intp nbins[2]
        nbins[0] = <np.npy_intp> self.thisptr.getNumParticles()
        nbins[1] = <np.npy_intp> self.thisptr.getNumBonds()
        cdef np.ndarray[np.uint32_t, ndim=2] result = \
            np.PyArray_SimpleNewFromData(
                2, nbins, np.NPY_UINT32, <void*> bonds)
        return result

    def getBonds(self):
        warnings.warn("The getBonds function is deprecated in favor "
                      "of the bonds class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.bonds

    @property
    def box(self):
        return freud.box.BoxFromCPP(self.thisptr.getBox())

    def getBox(self):
        warnings.warn("The getBox function is deprecated in favor "
                      "of the box class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.box

    @property
    def list_map(self):
        return self.thisptr.getListMap()

    def getListMap(self):
        warnings.warn("The getListMap function is deprecated in favor "
                      "of the list_map class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.list_map

    @property
    def rev_list_map(self):
        return self.thisptr.getRevListMap()

    def getRevListMap(self):
        warnings.warn("The getRevListMap function is deprecated in favor "
                      "of the rev_list_map class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.rev_list_map

cdef class BondingXYT:
    """Compute bonds in a 2D system using a
    (:math:`x`, :math:`y`, :math:`\\theta`) coordinate system.

    For each particle in the system determine which other particles are in
    which bonding sites.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    Args:
        x_max (float):
            Maximum :math:`x` distance at which to search for bonds.
        y_max (float):
            Maximum :math:`y` distance at which to search for bonds.
        bond_map (:class:`numpy.ndarray`):
            3D array containing the bond index for each :math:`x`, :math:`y`
            coordinate.
        bond_list (:class:`numpy.ndarray`):
            List containing the bond indices to be tracked,
            :code:`bond_list[i] = bond_index`.

    Attributes:
        bonds (:class:`numpy.ndarray`):
            Particle bonds.
        box (:py:class:`freud.box.Box`):
            Box used in the calculation.
        list_map (dict):
            The dict used to map bond index to list index.
        rev_list_map (dict):
            The dict used to map list idx to bond idx.
    """
    cdef freud._bond.BondingXYT * thisptr
    cdef rmax

    def __cinit__(self, float x_max, float y_max, bond_map, bond_list):
        # extract nr, nt from the bond_map
        n_t = bond_map.shape[0]
        n_y = bond_map.shape[1]
        n_x = bond_map.shape[2]
        n_bonds = bond_list.shape[0]
        bond_map = np.require(bond_map, requirements=["C"])
        bond_list = np.require(bond_list, requirements=["C"])
        cdef np.ndarray[uint, ndim=3] l_bond_map = bond_map
        cdef np.ndarray[uint, ndim=1] l_bond_list = bond_list
        self.thisptr = new freud._bond.BondingXYT(
            x_max, y_max, n_x, n_y, n_t, n_bonds,
            <unsigned int*> l_bond_map.data,
            <unsigned int*> l_bond_list.data)
        self.rmax = np.sqrt(x_max**2 + y_max**2)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, box, ref_points, ref_orientations, points=None,
                orientations=None, nlist=None):
        """Calculates the bonds.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points used to calculate the bonding.
            ref_orientations ((:math:`N_{particles}`, 1) or
            (:math:`N_{particles}`,) :class:`numpy.ndarray`):
                Reference orientations as angles to use in computation.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points used to calculate the bonding. Uses :code:`ref_points`
                if not provided or :code:`None`.
            orientations ((:math:`N_{particles}`, 1) or
            (:math:`N_{particles}`,) :class:`numpy.ndarray`, optional):
                Orientations as angles to use in computation. Uses
                :code:`ref_orientations` if not provided or :code:`None`.
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList to use to find bonds (Default value =
                :code:`None`).
        """
        cdef freud.box.Box b = freud.common.convert_box(box)

        if points is None:
            points = ref_points
        if orientations is None:
            orientations = ref_orientations

        ref_points = freud.common.convert_array(
            ref_points, 2, dtype=np.float32, contiguous=True,
            array_name="ref_points")
        if ref_points.shape[1] != 3:
            raise TypeError('ref_points should be an Nx3 array')

        ref_orientations = freud.common.convert_array(
            ref_orientations.squeeze(), 1, dtype=np.float32, contiguous=True,
            array_name="ref_orientations")

        points = freud.common.convert_array(
            points, 2, dtype=np.float32, contiguous=True, array_name="points")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        orientations = freud.common.convert_array(
            orientations.squeeze(), 1, dtype=np.float32, contiguous=True,
            array_name="orientations")

        defaulted_nlist = freud.locality.make_default_nlist(
            b, ref_points, points, self.rmax, nlist, None)
        cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]

        cdef np.ndarray[float, ndim=2] l_ref_points = ref_points
        cdef np.ndarray[float, ndim=1] l_ref_orientations = ref_orientations
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef np.ndarray[float, ndim=1] l_orientations = orientations
        cdef unsigned int n_ref = <unsigned int> ref_points.shape[0]
        cdef unsigned int n_p = <unsigned int> points.shape[0]
        with nogil:
            self.thisptr.compute(
                dereference(b.thisptr), nlist_.get_ptr(),
                <vec3[float]*> l_ref_points.data,
                <float*> l_ref_orientations.data, n_ref,
                <vec3[float]*> l_points.data,
                <float*> l_orientations.data, n_p)
        return self

    @property
    def bonds(self):
        cdef unsigned int * bonds = self.thisptr.getBonds().get()
        cdef np.npy_intp nbins[2]
        nbins[0] = <np.npy_intp> self.thisptr.getNumParticles()
        nbins[1] = <np.npy_intp> self.thisptr.getNumBonds()
        cdef np.ndarray[np.uint32_t, ndim=2] result = \
            np.PyArray_SimpleNewFromData(
                2, nbins, np.NPY_UINT32, <void*> bonds)
        return result

    def getBonds(self):
        warnings.warn("The getBonds function is deprecated in favor "
                      "of the bonds class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.bonds

    @property
    def box(self):
        return freud.box.BoxFromCPP(self.thisptr.getBox())

    def getBox(self):
        warnings.warn("The getBox function is deprecated in favor "
                      "of the box class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.box

    @property
    def list_map(self):
        return self.thisptr.getListMap()

    def getListMap(self):
        warnings.warn("The getListMap function is deprecated in favor "
                      "of the list_map class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.list_map

    @property
    def rev_list_map(self):
        return self.thisptr.getRevListMap()

    def getRevListMap(self):
        warnings.warn("The getRevListMap function is deprecated in favor "
                      "of the rev_list_map class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.rev_list_map

cdef class BondingXYZ:
    """Compute bonds in a 3D system using a
    (:math:`x`, :math:`y`, :math:`z`) coordinate system.

    For each particle in the system determine which other particles are in
    which bonding sites.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    Args:
        x_max (float):
            Maximum :math:`x` distance at which to search for bonds.
        y_max (float):
            Maximum :math:`y` distance at which to search for bonds.
        z_max (float):
            Maximum :math:`z` distance at which to search for bonds.
        bond_map (:class:`numpy.ndarray`):
            3D array containing the bond index for each :math:`x`, :math:`y`,
            :math:`z` coordinate.
        bond_list (:class:`numpy.ndarray`):
            List containing the bond indices to be tracked,
            :code:`bond_list[i] = bond_index`.

    Attributes:
        bonds (:class:`numpy.ndarray`):
            Particle bonds.
        box (:py:class:`freud.box.Box`):
            Box used in the calculation.
        list_map (dict):
            The dict used to map bond index to list index.
        rev_list_map (dict):
            The dict used to map list idx to bond idx.
    """
    cdef freud._bond.BondingXYZ * thisptr
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
        cdef np.ndarray[uint, ndim=3] l_bond_map = bond_map
        cdef np.ndarray[uint, ndim=1] l_bond_list = bond_list
        self.thisptr = new freud._bond.BondingXYZ(
            x_max, y_max, z_max, n_x, n_y, n_z, n_bonds,
            <unsigned int*> l_bond_map.data,
            <unsigned int*> l_bond_list.data)
        self.rmax = np.sqrt(x_max**2 + y_max**2 + z_max**2)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, box, ref_points, ref_orientations, points=None,
                orientations=None, nlist=None):
        """Calculates the bonds.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points used to calculate the bonding.
            ref_orientations ((:math:`N_{particles}`, 4) \
:class:`numpy.ndarray`):
                Reference orientations as quaternions to use in computation.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points used to calculate the bonding. Uses :code:`ref_points`
                if not provided or :code:`None`.
            orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`):
                Orientations as quaternions to use in computation. Uses
                :code:`ref_orientations` if not provided or :code:`None`.
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList to use to find bonds (Default value =
                :code:`None`).
        """
        cdef freud.box.Box b = freud.common.convert_box(box)

        if points is None:
            points = ref_points
        if orientations is None:
            orientations = ref_orientations

        ref_points = freud.common.convert_array(
            ref_points, 2, dtype=np.float32, contiguous=True,
            array_name="ref_points")
        if ref_points.shape[1] != 3:
            raise TypeError('ref_points should be an Nx3 array')

        ref_orientations = freud.common.convert_array(
            ref_orientations, 2, dtype=np.float32, contiguous=True,
            array_name="ref_orientations")
        if ref_orientations.shape[1] != 4:
            raise ValueError(
                "The 2nd dimension must have 4 values: q0, q1, q2, q3")

        points = freud.common.convert_array(
            points, 2, dtype=np.float32, contiguous=True, array_name="points")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        orientations = freud.common.convert_array(
            orientations, 2, dtype=np.float32, contiguous=True,
            array_name="orientations")
        if orientations.shape[1] != 4:
            raise ValueError(
                "The 2nd dimension must have 4 values: q0, q1, q2, q3")

        defaulted_nlist = freud.locality.make_default_nlist(
            b, ref_points, points, self.rmax, nlist, None)
        cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]

        cdef np.ndarray[float, ndim=2] l_ref_points = ref_points
        cdef np.ndarray[float, ndim=2] l_ref_orientations = ref_orientations
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef np.ndarray[float, ndim=2] l_orientations = orientations
        cdef unsigned int n_ref = <unsigned int> ref_points.shape[0]
        cdef unsigned int n_p = <unsigned int> points.shape[0]
        with nogil:
            self.thisptr.compute(
                dereference(b.thisptr), nlist_.get_ptr(),
                <vec3[float]*> l_ref_points.data,
                <quat[float]*> l_ref_orientations.data,
                n_ref,
                <vec3[float]*> l_points.data,
                <quat[float]*> l_orientations.data,
                n_p)
        return self

    @property
    def bonds(self):
        cdef unsigned int * bonds = self.thisptr.getBonds().get()
        cdef np.npy_intp nbins[2]
        nbins[0] = <np.npy_intp> self.thisptr.getNumParticles()
        nbins[1] = <np.npy_intp> self.thisptr.getNumBonds()
        cdef np.ndarray[np.uint32_t, ndim=2] result = \
            np.PyArray_SimpleNewFromData(
                2, nbins, np.NPY_UINT32, <void*> bonds)
        return result

    def getBonds(self):
        warnings.warn("The getBonds function is deprecated in favor "
                      "of the bonds class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.bonds

    @property
    def box(self):
        return freud.box.BoxFromCPP(self.thisptr.getBox())

    def getBox(self):
        warnings.warn("The getBox function is deprecated in favor "
                      "of the box class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.box

    @property
    def list_map(self):
        return self.thisptr.getListMap()

    def getListMap(self):
        warnings.warn("The getListMap function is deprecated in favor "
                      "of the list_map class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.list_map

    @property
    def rev_list_map(self):
        return self.thisptr.getRevListMap()

    def getRevListMap(self):
        warnings.warn("The getRevListMap function is deprecated in favor "
                      "of the rev_list_map class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.rev_list_map
