
from freud.util._VectorMath cimport vec3
from freud.util._VectorMath cimport quat
cimport freud._box as _box
cimport freud._bond as bond
from libcpp.map cimport map
import numpy as np
cimport numpy as np

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef class BondingAnalysis:
    """Analyze the bonds as calculated by Freud's Bonding modules.

    Determines the bond lifetimes and flux present in the system.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    :param r_max: distance to search for bonds
    :param bond_map: 3D array containing the bond index for each r, t2, t1 coordinate
    :param bond_list: list containing the bond indices to be tracked bond_list[i] = bond_index
    :type r_max: float
    :type bond_map: np.ndarray(shape=(n_r, n_t2, n_t1), dtype=np.uint32)
    :type bond_list: np.ndarray(shape=(n_bonds), dtype=np.uint32)
    """
    cdef bond.BondingAnalysis *thisptr

    def __cinit__(self, num_particles, num_bonds):
        self.thisptr = new bond.BondingAnalysis(num_particles, num_bonds)

    def __dealloc__(self):
        del self.thisptr

    # def compute(self, box, np.ndarray[float, ndim=2] ref_points, np.ndarray[float, ndim=1] ref_orientations,
    #     np.ndarray[float, ndim=2] points, np.ndarray[float, ndim=1] orientations):
    #     """
    #     Calculates the correlation function and adds to the current histogram.

    #     :param box: simulation box
    #     :param points: points to calculate the bonding
    #     :param orientations: orientations as angles to use in computation
    #     :type box: :py:meth:`freud.box.Box`
    #     :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
    #     :type orientations: np.ndarray(shape=(N), dtype=np.float32)
    #     """
    #     if ((points.dtype != np.float32) or (ref_points.dtype != np.float32)):
    #         raise ValueError("points must be a numpy float32 array")
    #     if ((points.ndim != 2) or (ref_points.ndim != 2)):
    #         raise ValueError("points must be a 2 dimensional array")
    #     if ((points.shape[1] != 3) or (ref_points.shape[1] != 3)):
    #         raise ValueError("the 2nd dimension must have 3 values: x, y, z")
    #     if ((orientations.dtype != np.float32) or (ref_orientations.dtype != np.float32)):
    #         raise ValueError("values must be a numpy float32 array")
    #     if ((orientations.ndim != 1) or (ref_orientations.ndim != 1)):
    #         raise ValueError("values must be a 1 dimensional array")
    #     cdef np.ndarray[float, ndim=2] l_ref_points = ref_points
    #     cdef np.ndarray[float, ndim=1] l_ref_orientations = ref_orientations
    #     cdef np.ndarray[float, ndim=2] l_points = points
    #     cdef np.ndarray[float, ndim=1] l_orientations = orientations
    #     cdef unsigned int n_ref = <unsigned int> ref_points.shape[0]
    #     cdef unsigned int n_p = <unsigned int> points.shape[0]
    #     cdef _box.Box l_box = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
    #         box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
    #     with nogil:
    #         self.thisptr.compute(l_box, <vec3[float]*> l_ref_points.data, <float*> l_ref_orientations.data, n_ref,
    #             <vec3[float]*> l_points.data, <float*> l_orientations.data, n_p)

    # def getBonds(self):
    #     """
    #     :return: particle bonds
    #     :rtype: np.ndarray(shape=(n_p, n_bonds), dtype=np.uint32)
    #     """
    #     cdef unsigned int *bonds = self.thisptr.getBonds().get()
    #     cdef np.npy_intp nbins[2]
    #     nbins[0] = <np.npy_intp>self.thisptr.getNumParticles()
    #     nbins[1] = <np.npy_intp>self.thisptr.getNumBonds()
    #     cdef np.ndarray[np.uint32_t, ndim=2] result = np.PyArray_SimpleNewFromData(2, nbins, np.NPY_UINT32,<void*>bonds)
    #     return result

    # def getBox(self):
    #     """
    #     Get the box used in the calculation

    #     :return: Freud Box
    #     :rtype: :py:meth:`freud.box.Box()`
    #     """
    #     return BoxFromCPP(<box.Box> self.thisptr.getBox())

    # def getListMap(self):
    #     """
    #     Get the dict used to map list idx to bond idx

    #     :return: list_map
    #     :rtype: dict()

    #     >>> list_idx = list_map[bond_idx]
    #     """
    #     return self.thisptr.getListMap()

    # def getRevListMap(self):
    #     """
    #     Get the dict used to map list idx to bond idx

    #     :return: list_map
    #     :rtype: dict()

    #     >>> bond_idx = list_map[list_idx]
    #     """
    #     return self.thisptr.getRevListMap()

cdef class BondingR12:
    """Compute the bonds each particle in the system.

    For each particle in the system determine which other particles are in which entropic bonding sites.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    :param r_max: distance to search for bonds
    :param bond_map: 3D array containing the bond index for each r, t2, t1 coordinate
    :param bond_list: list containing the bond indices to be tracked bond_list[i] = bond_index
    :type r_max: float
    :type bond_map: np.ndarray(shape=(n_r, n_t2, n_t1), dtype=np.uint32)
    :type bond_list: np.ndarray(shape=(n_bonds), dtype=np.uint32)
    """
    cdef bond.BondingR12 *thisptr

    def __cinit__(self, float r_max, np.ndarray[uint, ndim=3] bond_map, np.ndarray[uint, ndim=1] bond_list):
        # extract nr, nt from the bond_map
        n_r = bond_map.shape[0]
        n_t2 = bond_map.shape[1]
        n_t1 = bond_map.shape[2]
        n_bonds = bond_list.shape[0]
        cdef np.ndarray[uint, ndim=3] l_bond_map = bond_map
        cdef np.ndarray[uint, ndim=1] l_bond_list = bond_list
        self.thisptr = new bond.BondingR12(r_max, n_r, n_t2, n_t1, n_bonds,
            <unsigned int*>l_bond_map.data, <unsigned int*>l_bond_list.data)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, box, np.ndarray[float, ndim=2] ref_points, np.ndarray[float, ndim=1] ref_orientations,
        np.ndarray[float, ndim=2] points, np.ndarray[float, ndim=1] orientations):
        """
        Calculates the correlation function and adds to the current histogram.

        :param box: simulation box
        :param points: points to calculate the bonding
        :param orientations: orientations as angles to use in computation
        :type box: :py:meth:`freud.box.Box`
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type orientations: np.ndarray(shape=(N), dtype=np.float32)
        """
        if ((points.dtype != np.float32) or (ref_points.dtype != np.float32)):
            raise ValueError("points must be a numpy float32 array")
        if ((points.ndim != 2) or (ref_points.ndim != 2)):
            raise ValueError("points must be a 2 dimensional array")
        if ((points.shape[1] != 3) or (ref_points.shape[1] != 3)):
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        if ((orientations.dtype != np.float32) or (ref_orientations.dtype != np.float32)):
            raise ValueError("values must be a numpy float32 array")
        if ((orientations.ndim != 1) or (ref_orientations.ndim != 1)):
            raise ValueError("values must be a 1 dimensional array")
        cdef np.ndarray[float, ndim=2] l_ref_points = ref_points
        cdef np.ndarray[float, ndim=1] l_ref_orientations = ref_orientations
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef np.ndarray[float, ndim=1] l_orientations = orientations
        cdef unsigned int n_ref = <unsigned int> ref_points.shape[0]
        cdef unsigned int n_p = <unsigned int> points.shape[0]
        cdef _box.Box l_box = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
            box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.compute(l_box, <vec3[float]*> l_ref_points.data, <float*> l_ref_orientations.data, n_ref,
                <vec3[float]*> l_points.data, <float*> l_orientations.data, n_p)

    def getBonds(self):
        """
        :return: particle bonds
        :rtype: np.ndarray(shape=(n_p, n_bonds), dtype=np.uint32)
        """
        cdef unsigned int *bonds = self.thisptr.getBonds().get()
        cdef np.npy_intp nbins[2]
        nbins[0] = <np.npy_intp>self.thisptr.getNumParticles()
        nbins[1] = <np.npy_intp>self.thisptr.getNumBonds()
        cdef np.ndarray[np.uint32_t, ndim=2] result = np.PyArray_SimpleNewFromData(2, nbins, np.NPY_UINT32,<void*>bonds)
        return result

    def getBox(self):
        """
        Get the box used in the calculation

        :return: Freud Box
        :rtype: :py:meth:`freud.box.Box()`
        """
        return BoxFromCPP(<box.Box> self.thisptr.getBox())

    def getListMap(self):
        """
        Get the dict used to map list idx to bond idx

        :return: list_map
        :rtype: dict()

        >>> list_idx = list_map[bond_idx]
        """
        return self.thisptr.getListMap()

    def getRevListMap(self):
        """
        Get the dict used to map list idx to bond idx

        :return: list_map
        :rtype: dict()

        >>> bond_idx = list_map[list_idx]
        """
        return self.thisptr.getRevListMap()

cdef class BondingXY2D:
    """Compute the bonds each particle in the system.

    For each particle in the system determine which other particles are in which entropic bonding sites.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    :param x_max: maximum x distance at which to search for bonds
    :param y_max: maximum y distance at which to search for bonds
    :param bond_map: 3D array containing the bond index for each x, y coordinate
    :param bond_list: list containing the bond indices to be tracked bond_list[i] = bond_index
    :type x_max: float
    :type y_max: float
    :type bond_map: np.ndarray(shape=(n_y, n_x), dtype=np.uint32)
    :type bond_list: np.ndarray(shape=(n_bonds), dtype=np.uint32)
    """
    cdef bond.BondingXY2D *thisptr

    def __cinit__(self, float x_max, float y_max, np.ndarray[uint, ndim=2] bond_map,
        np.ndarray[uint, ndim=1] bond_list):
        # extract nr, nt from the bond_map
        n_y = bond_map.shape[0]
        n_x = bond_map.shape[1]
        n_bonds = bond_list.shape[0]
        cdef np.ndarray[uint, ndim=2] l_bond_map = bond_map
        cdef np.ndarray[uint, ndim=1] l_bond_list = bond_list
        self.thisptr = new bond.BondingXY2D(x_max, y_max, n_x, n_y, n_bonds,
            <unsigned int*>l_bond_map.data, <unsigned int*>l_bond_list.data)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, box, np.ndarray[float, ndim=2] ref_points, np.ndarray[float, ndim=1] ref_orientations,
        np.ndarray[float, ndim=2] points, np.ndarray[float, ndim=1] orientations):
        """
        Calculates the correlation function and adds to the current histogram.

        :param box: simulation box
        :param points: points to calculate the bonding
        :param orientations: orientations as angles to use in computation
        :type box: :py:meth:`freud.box.Box`
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type orientations: np.ndarray(shape=(N), dtype=np.float32)
        """
        if ((points.dtype != np.float32) or (ref_points.dtype != np.float32)):
            raise ValueError("points must be a numpy float32 array")
        if ((points.ndim != 2) or (ref_points.ndim != 2)):
            raise ValueError("points must be a 2 dimensional array")
        if ((points.shape[1] != 3) or (ref_points.shape[1] != 3)):
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        if ((orientations.dtype != np.float32) or (ref_orientations.dtype != np.float32)):
            raise ValueError("values must be a numpy float32 array")
        if ((orientations.ndim != 1) or (ref_orientations.ndim != 1)):
            raise ValueError("values must be a 1 dimensional array")
        cdef np.ndarray[float, ndim=2] l_ref_points = ref_points
        cdef np.ndarray[float, ndim=1] l_ref_orientations = ref_orientations
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef np.ndarray[float, ndim=1] l_orientations = orientations
        cdef unsigned int n_ref = <unsigned int> ref_points.shape[0]
        cdef unsigned int n_p = <unsigned int> points.shape[0]
        cdef _box.Box l_box = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
            box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.compute(l_box, <vec3[float]*> l_ref_points.data, <float*> l_ref_orientations.data, n_ref,
                <vec3[float]*> l_points.data, <float*> l_orientations.data, n_p)

    def getBonds(self):
        """
        :return: particle bonds
        :rtype: np.ndarray(shape=(n_p, n_bonds), dtype=np.uint32)
        """
        cdef unsigned int *bonds = self.thisptr.getBonds().get()
        cdef np.npy_intp nbins[2]
        nbins[0] = <np.npy_intp>self.thisptr.getNumParticles()
        nbins[1] = <np.npy_intp>self.thisptr.getNumBonds()
        cdef np.ndarray[np.uint32_t, ndim=2] result = np.PyArray_SimpleNewFromData(2, nbins, np.NPY_UINT32,<void*>bonds)
        return result

    def getBox(self):
        """
        Get the box used in the calculation

        :return: Freud Box
        :rtype: :py:meth:`freud.box.Box()`
        """
        return BoxFromCPP(<box.Box> self.thisptr.getBox())

    def getListMap(self):
        """
        Get the dict used to map list idx to bond idx

        :return: list_map
        :rtype: dict()

        >>> list_idx = list_map[bond_idx]
        """
        return self.thisptr.getListMap()

    def getRevListMap(self):
        """
        Get the dict used to map list idx to bond idx

        :return: list_map
        :rtype: dict()

        >>> bond_idx = list_map[list_idx]
        """
        return self.thisptr.getRevListMap()

cdef class BondingXYT:
    """Compute the bonds each particle in the system.

    For each particle in the system determine which other particles are in which entropic bonding sites.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    :param x_max: maximum x distance at which to search for bonds
    :param y_max: maximum y distance at which to search for bonds
    :param bond_map: 3D array containing the bond index for each x, y coordinate
    :param bond_list: list containing the bond indices to be tracked bond_list[i] = bond_index
    :type x_max: float
    :type y_max: float
    :type bond_map: np.ndarray(shape=(n_t, n_y, n_x), dtype=np.uint32)
    :type bond_list: np.ndarray(shape=(n_bonds), dtype=np.uint32)
    """
    cdef bond.BondingXYT *thisptr

    def __cinit__(self, float x_max, float y_max, np.ndarray[uint, ndim=3] bond_map,
        np.ndarray[uint, ndim=1] bond_list):
        # extract nr, nt from the bond_map
        n_t = bond_map.shape[0]
        n_y = bond_map.shape[1]
        n_x = bond_map.shape[2]
        n_bonds = bond_list.shape[0]
        cdef np.ndarray[uint, ndim=3] l_bond_map = bond_map
        cdef np.ndarray[uint, ndim=1] l_bond_list = bond_list
        self.thisptr = new bond.BondingXYT(x_max, y_max, n_x, n_y, n_t, n_bonds, <unsigned int*>l_bond_map.data,
            <unsigned int*>l_bond_list.data)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, box, np.ndarray[float, ndim=2] ref_points, np.ndarray[float, ndim=1] ref_orientations,
        np.ndarray[float, ndim=2] points, np.ndarray[float, ndim=1] orientations):
        """
        Calculates the correlation function and adds to the current histogram.

        :param box: simulation box
        :param points: points to calculate the bonding
        :param orientations: orientations as angles to use in computation
        :type box: :py:meth:`freud.box.Box`
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type orientations: np.ndarray(shape=(N), dtype=np.float32)
        """
        if ((points.dtype != np.float32) or (ref_points.dtype != np.float32)):
            raise ValueError("points must be a numpy float32 array")
        if ((points.ndim != 2) or (ref_points.ndim != 2)):
            raise ValueError("points must be a 2 dimensional array")
        if ((points.shape[1] != 3) or (ref_points.shape[1] != 3)):
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        if ((orientations.dtype != np.float32) or (ref_orientations.dtype != np.float32)):
            raise ValueError("values must be a numpy float32 array")
        if ((orientations.ndim != 1) or (ref_orientations.ndim != 1)):
            raise ValueError("values must be a 1 dimensional array")
        cdef np.ndarray[float, ndim=2] l_ref_points = ref_points
        cdef np.ndarray[float, ndim=1] l_ref_orientations = ref_orientations
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef np.ndarray[float, ndim=1] l_orientations = orientations
        cdef unsigned int n_ref = <unsigned int> ref_points.shape[0]
        cdef unsigned int n_p = <unsigned int> points.shape[0]
        cdef _box.Box l_box = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
            box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.compute(l_box, <vec3[float]*> l_ref_points.data, <float*> l_ref_orientations.data, n_ref,
                <vec3[float]*> l_points.data, <float*> l_orientations.data, n_p)

    def getBonds(self):
        """
        :return: particle bonds
        :rtype: np.ndarray(shape=(n_p, n_bonds), dtype=np.uint32)
        """
        cdef unsigned int *bonds = self.thisptr.getBonds().get()
        cdef np.npy_intp nbins[2]
        nbins[0] = <np.npy_intp>self.thisptr.getNumParticles()
        nbins[1] = <np.npy_intp>self.thisptr.getNumBonds()
        cdef np.ndarray[np.uint32_t, ndim=2] result = np.PyArray_SimpleNewFromData(2, nbins, np.NPY_UINT32,<void*>bonds)
        return result

    def getBox(self):
        """
        Get the box used in the calculation

        :return: Freud Box
        :rtype: :py:meth:`freud.box.Box()`
        """
        return BoxFromCPP(<box.Box> self.thisptr.getBox())

    def getListMap(self):
        """
        Get the dict used to map list idx to bond idx

        :return: list_map
        :rtype: dict()

        >>> list_idx = list_map[bond_idx]
        """
        return self.thisptr.getListMap()

    def getRevListMap(self):
        """
        Get the dict used to map list idx to bond idx

        :return: list_map
        :rtype: dict()

        >>> bond_idx = list_map[list_idx]
        """
        return self.thisptr.getRevListMap()

cdef class BondingXYZ:
    """Compute the bonds each particle in the system.

    For each particle in the system determine which other particles are in which entropic bonding sites.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    :param x_max: maximum x distance at which to search for bonds
    :param y_max: maximum y distance at which to search for bonds
    :param z_max: maximum z distance at which to search for bonds
    :param bond_map: 3D array containing the bond index for each x, y, z coordinate
    :param bond_list: list containing the bond indices to be tracked bond_list[i] = bond_index
    :type x_max: float
    :type y_max: float
    :type z_max: float
    :type bond_map: np.ndarray(shape=(n_z, n_y, n_x), dtype=np.uint32)
    :type bond_list: np.ndarray(shape=(n_bonds), dtype=np.uint32)
    """
    cdef bond.BondingXYZ *thisptr

    def __cinit__(self, float x_max, float y_max, float z_max, np.ndarray[uint, ndim=3] bond_map,
        np.ndarray[uint, ndim=1] bond_list):
        # extract nr, nt from the bond_map
        n_z = bond_map.shape[0]
        n_y = bond_map.shape[1]
        n_x = bond_map.shape[2]
        n_bonds = bond_list.shape[0]
        cdef np.ndarray[uint, ndim=3] l_bond_map = bond_map
        cdef np.ndarray[uint, ndim=1] l_bond_list = bond_list
        self.thisptr = new bond.BondingXYZ(x_max, y_max, z_max, n_x, n_y, n_z, n_bonds, <unsigned int*>l_bond_map.data,
            <unsigned int*>l_bond_list.data)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, box, np.ndarray[float, ndim=2] ref_points, np.ndarray[float, ndim=2] ref_orientations,
        np.ndarray[float, ndim=2] points, np.ndarray[float, ndim=2] orientations):
        """
        Calculates the correlation function and adds to the current histogram.

        :param box: simulation box
        :param points: points to calculate the bonding
        :param orientations: orientations as angles to use in computation
        :type box: :py:meth:`freud.box.Box`
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type orientations: np.ndarray(shape=(N, 4), dtype=np.float32)
        """
        if ((points.dtype != np.float32) or (ref_points.dtype != np.float32)):
            raise ValueError("points must be a numpy float32 array")
        if ((points.ndim != 2) or (ref_points.ndim != 2)):
            raise ValueError("points must be a 2 dimensional array")
        if ((points.shape[1] != 3) or (ref_points.shape[1] != 3)):
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        if ((orientations.dtype != np.float32) or (ref_orientations.dtype != np.float32)):
            raise ValueError("values must be a numpy float32 array")
        if ((orientations.ndim != 2) or (ref_orientations.ndim != 2)):
            raise ValueError("values must be a 2 dimensional array")
        if ((orientations.shape[1] != 4) or (ref_orientations.shape[1] != 4)):
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=2] l_ref_points = ref_points
        cdef np.ndarray[float, ndim=2] l_ref_orientations = ref_orientations
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef np.ndarray[float, ndim=2] l_orientations = orientations
        cdef unsigned int n_ref = <unsigned int> ref_points.shape[0]
        cdef unsigned int n_p = <unsigned int> points.shape[0]
        cdef _box.Box l_box = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
            box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.compute(l_box, <vec3[float]*> l_ref_points.data, <quat[float]*> l_ref_orientations.data, n_ref,
                <vec3[float]*> l_points.data, <quat[float]*> l_orientations.data, n_p)

    def getBonds(self):
        """
        :return: particle bonds
        :rtype: np.ndarray(shape=(n_p, n_bonds), dtype=np.uint32)
        """
        cdef unsigned int *bonds = self.thisptr.getBonds().get()
        cdef np.npy_intp nbins[2]
        nbins[0] = <np.npy_intp>self.thisptr.getNumParticles()
        nbins[1] = <np.npy_intp>self.thisptr.getNumBonds()
        cdef np.ndarray[np.uint32_t, ndim=2] result = np.PyArray_SimpleNewFromData(2, nbins, np.NPY_UINT32,<void*>bonds)
        return result

    def getBox(self):
        """
        Get the box used in the calculation

        :return: Freud Box
        :rtype: :py:meth:`freud.box.Box()`
        """
        return BoxFromCPP(<box.Box> self.thisptr.getBox())

    def getListMap(self):
        """
        Get the dict used to map list idx to bond idx

        :return: list_map
        :rtype: dict()

        >>> list_idx = list_map[bond_idx]
        """
        return self.thisptr.getListMap()

    def getRevListMap(self):
        """
        Get the dict used to map list idx to bond idx

        :return: list_map
        :rtype: dict()

        >>> bond_idx = list_map[list_idx]
        """
        return self.thisptr.getRevListMap()
