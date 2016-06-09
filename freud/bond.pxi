
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

cdef class BondingR12:
    """Compute the bonds each particle in the system.

    For each particle in the system determine which other particles are in which entropic bonding sites.

    .. note:: currently being debugged. not guaranteed to work.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    :param r_max: distance to search for bonds
    :param bond_map: 2D array containing the bond index for each x, y coordinate
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

    def compute(self, box, np.ndarray[float, ndim=2] points, np.ndarray[float, ndim=1] orientations):
        """
        Calculates the correlation function and adds to the current histogram.

        :param box: simulation box
        :param points: points to calculate the bonding
        :param orientations: orientations as angles to use in computation
        :type box: :py:meth:`freud.box.Box`
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type orientations: np.ndarray(shape=(N), dtype=np.float32)
        """
        if points.dtype != np.float32:
            raise ValueError("points must be a numpy float32 array")
        if points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        if orientations.dtype != np.float32:
            raise ValueError("values must be a numpy float32 array")
        if orientations.ndim != 1:
            raise ValueError("values must be a 1 dimensional array")
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef np.ndarray[float, ndim=1] l_orientations = orientations
        cdef unsigned int n_p = <unsigned int> points.shape[0]
        cdef _box.Box l_box = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
            box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.compute(l_box, <vec3[float]*> l_points.data, <float*> l_orientations.data, n_p)

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
