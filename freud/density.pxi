# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from freud.util._VectorMath cimport vec3
from libcpp.memory cimport shared_ptr
from cython.operator cimport dereference
from libc.string cimport memcpy
import numpy as np
cimport freud._box as _box
cimport freud._locality as locality
cimport freud._density as density
cimport numpy as np

# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef class FloatCF:
    """Computes the pairwise correlation function :math:`\\left< p*q \\right>
    \\left( r \\right)` between two sets of points with associated values
    :math:`p` and :math:`q`.

    Two sets of points and two sets of real values associated with those
    points are given. Computing the correlation function results in an
    array of the expected (average) product of all values at a given
    radial distance.

    The values of :math:`r` where the correlation function is computed are
    controlled by the :code:`rmax` and :code:`dr` parameters to the
    constructor. :code:`rmax` determines the maximum distance at which to
    compute the correlation function and :code:`dr` is the step size for each
    bin.

    .. note::
        2D: :py:class:`freud.density.FloatCF` properly handles 2D boxes.
        The points must be passed in as :code:`[x, y, 0]`.
        Failing to set z=0 will lead to undefined behavior.

    Self-correlation: It is often the case that we wish to compute the
    correlation function of a set of points with itself. If given the same
    arrays for both :code:`points` and :code:`ref_points`, we omit
    accumulating the self-correlation value in the first bin.

    .. moduleauthor:: Matthew Spellings <mspells@umich.edu>

    Args:
        rmax (float):
            Distance over which to calculate.
        dr (float):
            Bin size.

    Attributes:
        RDF ((:math:`N_{bins}`) :class:`numpy.ndarray`):
            Expected (average) product of all values at a given radial
            distance.
        box (:py:class:`freud.box.Box`):
            Box used in the calculation.
        counts ((:math:`N_{bins}`) :class:`numpy.ndarray`):
            The counts of each histogram bin.
        R ((:math:`N_{bins}`) :class:`numpy.ndarray`):
            The values of bin centers.
    """
    cdef density.CorrelationFunction[double] * thisptr
    cdef rmax

    def __cinit__(self, float rmax, float dr):
        if dr <= 0.0:
            raise ValueError("dr must be > 0")
        self.thisptr = new density.CorrelationFunction[double](rmax, dr)
        self.rmax = rmax

    def __dealloc__(self):
        del self.thisptr

    def accumulate(self, box, ref_points, refValues, points, values,
                   nlist=None):
        """Calculates the correlation function and adds to the current
        histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points to calculate the local density.
            refValues ((:math:`N_{particles}`) :class:`numpy.ndarray`):
                Values to use in computation.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the bonding.
            values ((:math:`N_{particles}`):
                Values to use in computation.
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList to use to find bonds (Default value = None).
        """
        box = freud.common.convert_box(box)
        ref_points = freud.common.convert_array(
            ref_points, 2, dtype=np.float32, contiguous=True,
            array_name="ref_points")
        points = freud.common.convert_array(
            points, 2, dtype=np.float32, contiguous=True, array_name="points")
        refValues = freud.common.convert_array(
            refValues, 1, dtype=np.float64, contiguous=True)
        values = freud.common.convert_array(
            values, 1, dtype=np.float64, contiguous=True)
        if ref_points.shape[1] != 3 or points.shape[1] != 3:
            raise ValueError("The 2nd dimension must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=2] l_ref_points = ref_points
        cdef np.ndarray[float, ndim=2] l_points
        if ref_points is points:
            l_points = l_ref_points
        else:
            l_points = points
        cdef np.ndarray[np.float64_t, ndim=1] l_refValues = refValues
        cdef np.ndarray[np.float64_t, ndim=1] l_values
        if values is refValues:
            l_values = l_refValues
        else:
            l_values = values

        defaulted_nlist = make_default_nlist(
            box, ref_points, points, self.rmax, nlist, None)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        cdef locality.NeighborList * nlist_ptr = nlist_.get_ptr()

        cdef unsigned int n_ref = <unsigned int> ref_points.shape[0]
        cdef unsigned int n_p = <unsigned int> points.shape[0]
        cdef _box.Box l_box = _box.Box(
            box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
            box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.accumulate(
                l_box, nlist_ptr,
                <vec3[float]*> l_ref_points.data,
                <double*> l_refValues.data, n_ref,
                <vec3[float]*> l_points.data,
                <double*> l_values.data,
                n_p)
        return self

    @property
    def RDF(self):
        return self.getRDF()

    def getRDF(self):
        """Returns the radial distribution function.

        Returns:
            (:math:`N_{bins}`) :class:`numpy.ndarray`:
                Expected (average) product of all values at a given radial
                distance.
        """
        cdef shared_ptr[double] rdf_ptr = self.thisptr.getRDF()
        cdef double * rdf = rdf_ptr.get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.thisptr.getNBins()
        cdef np.ndarray[np.float64_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(
                1, nbins, np.NPY_FLOAT64, <void*> rdf)
        return result

    @property
    def box(self):
        return self.getBox()

    def getBox(self):
        """Get the box used in the calculation.

        Returns:
            :py:class:`freud.box.Box`: freud Box.
        """
        return BoxFromCPP(<box.Box> self.thisptr.getBox())

    def resetCorrelationFunction(self):
        """Resets the values of the correlation function histogram in
        memory.
        """
        self.thisptr.reset()

    def compute(self, box, ref_points, refValues, points, values, nlist=None):
        """Calculates the correlation function for the given points. Will
        overwrite the current histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points to calculate the local density.
            refValues ((:math:`N_{particles}`) :class:`numpy.ndarray`):
                Values to use in computation.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the local density.
            values ((:math:`N_{particles}`):
                Values to use in computation.
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList to use to find bonds (Default value = None).
        """
        self.thisptr.reset()
        self.accumulate(box, ref_points, refValues, points, values, nlist)
        return self

    def reduceCorrelationFunction(self):
        """Reduces the histogram in the values over N processors to a single
        histogram. This is called automatically by
        :py:meth:`freud.density.FloatCF.getRDF()`,
        :py:meth:`freud.density.FloatCF.getCounts()`.
        """
        self.thisptr.reduceCorrelationFunction()

    @property
    def counts(self):
        return self.getCounts()

    def getCounts(self):
        """Get counts of each histogram bin.

        Returns:
            (:math:`N_{bins}`) :class:`numpy.ndarray`:
                Counts of each histogram bin.
        """
        cdef unsigned int * counts = self.thisptr.getCounts().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.thisptr.getNBins()
        cdef np.ndarray[np.uint32_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(
                1, nbins, np.NPY_UINT32, <void*> counts)
        return result

    @property
    def R(self):
        return self.getR()

    def getR(self):
        """Get bin centers.

        Returns:
            (:math:`N_{bins}`) :class:`numpy.ndarray`: Values of bin centers.
        """
        cdef float * r = self.thisptr.getR().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.thisptr.getNBins()
        cdef np.ndarray[np.float32_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(
                1, nbins, np.NPY_FLOAT32, <void*> r)
        return result

cdef class ComplexCF:
    """Computes the pairwise correlation function :math:`\\left< p*q \\right>
    \\left( r \\right)` between two sets of points with associated values
    :math:`p` and :math:`q`.

    Two sets of points and two sets of complex values associated with those
    points are given. Computing the correlation function results in an
    array of the expected (average) product of all values at a given
    radial distance.

    The values of :math:`r` where the correlation function is computed are
    controlled by the :code:`rmax` and :code:`dr` parameters to the
    constructor. :code:`rmax` determines the maximum distance at which to
    compute the correlation function and :code:`dr` is the step size for each
    bin.

    .. note::
        2D: :py:class:`freud.density.ComplexCF` properly handles 2D boxes.
        The points must be passed in as :code:`[x, y, 0]`.
        Failing to set z=0 will lead to undefined behavior.

    Self-correlation: It is often the case that we wish to compute the
    correlation function of a set of points with itself. If given the same
    arrays for both :code:`points` and :code:`ref_points`, we omit
    accumulating the self-correlation value in the first bin.

    .. moduleauthor:: Matthew Spellings <mspells@umich.edu>

    Args:
        rmax (float): Distance over which to calculate.
        dr (float): Bin size.

    Attributes:
        RDF ((:math:`N_{bins}`) :class:`numpy.ndarray`):
            Expected (average) product of all values at a given radial
            distance.
        box (:py:class:`freud.box.Box`):
            Box used in the calculation.
        counts ((:math:`N_{bins}`) :class:`numpy.ndarray`):
            The counts of each histogram bin.
        R ((:math:`N_{bins}`) :class:`numpy.ndarray`):
            The values of bin centers.
    """
    cdef density.CorrelationFunction[np.complex128_t] * thisptr
    cdef rmax

    def __cinit__(self, float rmax, float dr):
        if dr <= 0.0:
            raise ValueError("dr must be > 0")
        self.thisptr = new density.CorrelationFunction[np.complex128_t](
            rmax, dr)
        self.rmax = rmax

    def __dealloc__(self):
        del self.thisptr

    def accumulate(self, box, ref_points, refValues, points, values,
                   nlist=None):
        """Calculates the correlation function and adds to the current
        histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points to calculate the local density.
            refValues ((:math:`N_{particles}`) :class:`numpy.ndarray`):
                Values to use in computation.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the bonding.
            values ((:math:`N_{particles}`):
                Values to use in computation.
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList to use to find bonds (Default value = None).
        """
        box = freud.common.convert_box(box)
        ref_points = freud.common.convert_array(
            ref_points, 2, dtype=np.float32, contiguous=True,
            array_name="ref_points")
        points = freud.common.convert_array(
            points, 2, dtype=np.float32, contiguous=True, array_name="points")
        refValues = freud.common.convert_array(
            refValues, 1, dtype=np.complex128, contiguous=True)
        values = freud.common.convert_array(
            values, 1, dtype=np.complex128, contiguous=True)
        if ref_points.shape[1] != 3 or points.shape[1] != 3:
            raise ValueError("The 2nd dimension must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=2] l_ref_points = ref_points
        cdef np.ndarray[float, ndim=2] l_points
        if ref_points is points:
            l_points = l_ref_points
        else:
            l_points = points
        cdef np.ndarray[np.complex128_t, ndim=1] l_refValues = refValues
        cdef np.ndarray[np.complex128_t, ndim=1] l_values
        if values is refValues:
            l_values = l_refValues
        else:
            l_values = values

        defaulted_nlist = make_default_nlist(
            box, ref_points, points, self.rmax, nlist, None)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        cdef locality.NeighborList * nlist_ptr = nlist_.get_ptr()

        cdef unsigned int n_ref = <unsigned int> ref_points.shape[0]
        cdef unsigned int n_p = <unsigned int> points.shape[0]
        cdef _box.Box l_box = _box.Box(
            box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
            box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.accumulate(
                l_box, nlist_ptr,
                <vec3[float]*> l_ref_points.data,
                <np.complex128_t*> l_refValues.data,
                n_ref,
                <vec3[float]*> l_points.data,
                <np.complex128_t*> l_values.data,
                n_p)
        return self

    @property
    def RDF(self):
        return self.getRDF()

    def getRDF(self):
        """Get the RDF.

        Returns:
            (:math:`N_{bins}`) :class:`numpy.ndarray`:
                Expected (average) product of all values at a given radial
                distance.
        """
        cdef shared_ptr[np.complex128_t] rdf_ptr = self.thisptr.getRDF()
        cdef np.complex128_t * rdf = rdf_ptr.get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.thisptr.getNBins()
        cdef np.ndarray[np.complex128_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(
                1, nbins, np.NPY_COMPLEX128, <void*> rdf)
        return result

    @property
    def box(self):
        return self.getBox()

    def getBox(self):
        """Get the box used in the calculations.

        Returns:
          :class:`freud.box.Box`: freud Box.
        """
        return BoxFromCPP(<box.Box> self.thisptr.getBox())

    def resetCorrelationFunction(self):
        """Resets the values of the correlation function histogram in
        memory.
        """
        self.thisptr.reset()

    def compute(self, box, ref_points, refValues, points, values, nlist=None):
        """Calculates the correlation function for the given points. Will
        overwrite the current histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points to calculate the local density.
            refValues ((:math:`N_{particles}`) :class:`numpy.ndarray`):
                Values to use in computation.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the bonding.
            values ((:math:`N_{particles}`):
                Values to use in computation.
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList to use to find bonds (Default value = None)
        """
        self.thisptr.reset()
        self.accumulate(box, ref_points, refValues, points, values, nlist)
        return self

    def reduceCorrelationFunction(self):
        """Reduces the histogram in the values over N processors to a single
        histogram. This is called automatically by
        :py:meth:`freud.density.ComplexCF.getRDF()`,
        :py:meth:`freud.density.ComplexCF.getCounts()`.
        """
        self.thisptr.reduceCorrelationFunction()

    @property
    def counts(self):
        return self.getCounts()

    def getCounts(self):
        """Get the counts of each histogram bin.

        Returns:
            (:math:`N_{bins}`) :class:`numpy.ndarray`:
                Counts of each histogram bin.
        """
        cdef unsigned int * counts = self.thisptr.getCounts().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.thisptr.getNBins()
        cdef np.ndarray[np.uint32_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(
                1, nbins, np.NPY_UINT32, <void*> counts)
        return result

    @property
    def R(self):
        return self.getR()

    def getR(self):
        """Get The value of bin centers.

        Returns:
            (:math:`N_{bins}`) :class:`numpy.ndarray`: Values of bin centers.
        """
        cdef float * r = self.thisptr.getR().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.thisptr.getNBins()
        cdef np.ndarray[np.float32_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(
                1, nbins, np.NPY_FLOAT32, <void*> r)
        return result

cdef class GaussianDensity:
    """Computes the density of a system on a grid.

    Replaces particle positions with a Gaussian blur and calculates the
    contribution from the grid based upon the distance of the grid cell from
    the center of the Gaussian. The dimensions of the image (grid) are set in
    the constructor, and can either be set equally for all dimensions or for
    each dimension independently.

    - Constructor Calls:

        Initialize with all dimensions identical::

            freud.density.GaussianDensity(width, r_cut, dr)

        Initialize with each dimension specified::

            freud.density.GaussianDensity(width_x, width_y, width_z, r_cut, dr)

    .. moduleauthor:: Joshua Anderson <joaander@umich.edu>

    Args:
        width (unsigned int):
            Number of pixels to make the image.
        width_x (unsigned int):
            Number of pixels to make the image in x.
        width_y (unsigned int):
            Number of pixels to make the image in y.
        width_z (unsigned int):
            Number of pixels to make the image in z.
        r_cut (float):
            Distance over which to blur.
        sigma (float):
            Sigma parameter for Gaussian.

    Attributes:
        box (:py:class:`freud.box.Box`):
            Box used in the calculation.
        gaussian_density ((:math:`w_x`, :math:`w_y`, :math:`w_z`) \
        :class:`numpy.ndarray`):
            The image grid with the Gaussian density.
        counts ((:math:`N_{bins}`) :class:`numpy.ndarray`):
            The counts of each histogram bin.
        R ((:math:`N_{bins}`) :class:`numpy.ndarray`):
            The values of bin centers.
    """
    cdef density.GaussianDensity * thisptr

    def __cinit__(self, *args):
        if len(args) == 3:
            self.thisptr = new density.GaussianDensity(
                args[0], args[1], args[2])
        elif len(args) == 5:
            self.thisptr = new density.GaussianDensity(
                args[0], args[1], args[2], args[3], args[4])
        else:
            raise TypeError('GaussianDensity takes exactly 3 or 5 arguments')

    @property
    def box(self):
        return self.getBox()

    def getBox(self):
        """Get the box used in the calculation.

        Returns:
            :class:`freud.box.Box`: freud Box.
        """
        return BoxFromCPP(self.thisptr.getBox())

    def compute(self, box, points):
        """Calculates the Gaussian blur for the specified points. Does not
        accumulate (will overwrite current image).

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the local density.
        """
        box = freud.common.convert_box(box)
        points = freud.common.convert_array(
            points, 2, dtype=np.float32, contiguous=True, array_name="points")
        if points.shape[1] != 3:
            raise ValueError("The 2nd dimension must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int n_p = points.shape[0]
        cdef _box.Box l_box = _box.Box(
            box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
            box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.compute(l_box, <vec3[float]*> l_points.data, n_p)
        return self

    @property
    def gaussian_density(self):
        return self.getGaussianDensity()

    def getGaussianDensity(self):
        """ Get the image grid with the Gaussian density.

        Returns:
            (:math:`w_x`, :math:`w_y`, :math:`w_z`) :class:`numpy.ndarray`:
                Image (grid) with values of Gaussian.
        """
        cdef float * density = self.thisptr.getDensity().get()
        cdef np.npy_intp nbins[1]
        arraySize = self.thisptr.getWidthY() * self.thisptr.getWidthX()
        cdef _box.Box l_box = self.thisptr.getBox()
        if not l_box.is2D():
            arraySize *= self.thisptr.getWidthZ()
        nbins[0] = <np.npy_intp> arraySize
        cdef np.ndarray[np.float32_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(
                1, nbins, np.NPY_FLOAT32, <void*> density)
        if l_box.is2D():
            arrayShape = (self.thisptr.getWidthY(),
                          self.thisptr.getWidthX())
        else:
            arrayShape = (self.thisptr.getWidthZ(),
                          self.thisptr.getWidthY(),
                          self.thisptr.getWidthX())
        pyResult = np.reshape(np.ascontiguousarray(result), arrayShape)
        return pyResult

    def resetDensity(self):
        """Resets the values of GaussianDensity in memory."""
        self.thisptr.reset()

cdef class LocalDensity:
    """ Computes the local density around a particle.

    The density of the local environment is computed and averaged for a given
    set of reference points in a sea of data points. Providing the same points
    calculates them against themselves. Computing the local density results in
    an array listing the value of the local density around each reference
    point. Also available is the number of neighbors for each reference point,
    giving the user the ability to count the number of particles in that
    region.

    The values to compute the local density are set in the constructor.
    :code:`r_cut` sets the maximum distance at which to calculate the local
    density. :code:`volume` is the volume of a single particle.
    :code:`diameter` is the diameter of the circumsphere of an individual
    particle.

    .. note::
        2D: :py:class:`freud.density.LocalDensity` properly handles 2D boxes.
        The points must be passed in as :code:`[x, y, 0]`.
        Failing to set z=0 will lead to undefined behavior.

    .. moduleauthor:: Joshua Anderson <joaander@umich.edu>

    Args:
        r_cut (float):
            Maximum distance over which to calculate the density.
        volume (float):
            Volume of a single particle.
        diameter (float):
            Diameter of particle circumsphere.

    Attributes:
        box (:py:class:`freud.box.Box`):
            Box used in the calculation.
        density ((:math:`N_{particles}`) :class:`numpy.ndarray`):
            Density per particle.
        num_neighbors ((:math:`N_{particles}`) :class:`numpy.ndarray`):
            Number of neighbors for each particle..
    """
    cdef density.LocalDensity * thisptr
    cdef r_cut
    cdef diameter

    def __cinit__(self, float r_cut, float volume, float diameter):
        self.thisptr = new density.LocalDensity(r_cut, volume, diameter)
        self.r_cut = r_cut
        self.diameter = diameter

    @property
    def box(self):
        return self.getBox()

    def getBox(self):
        """Get the box used in the calculation.

        Returns:
            :class:`freud.box.Box`: freud Box.
        """
        return BoxFromCPP(self.thisptr.getBox())

    def compute(self, box, ref_points, points=None, nlist=None):
        """Calculates the local density for the specified points. Does not
        accumulate (will overwrite current data).

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points to calculate the local density.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the local density.
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList to use to find bonds (Default value = None).
        """
        box = freud.common.convert_box(box)
        if points is None:
            points = ref_points
        ref_points = freud.common.convert_array(
            ref_points, 2, dtype=np.float32, contiguous=True,
            array_name="ref_points")
        points = freud.common.convert_array(
            points, 2, dtype=np.float32, contiguous=True, array_name="points")
        if ref_points.shape[1] != 3 or points.shape[1] != 3:
            raise ValueError("The 2nd dimension must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=2] l_ref_points = ref_points
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int n_ref = <unsigned int> ref_points.shape[0]
        cdef unsigned int n_p = <unsigned int> points.shape[0]

        cdef _box.Box l_box = _box.Box(
            box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
            box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())

        # local density of each particle includes itself (cutoff
        # distance is r_cut + diam/2 because of smoothing)
        defaulted_nlist = make_default_nlist(
            box, ref_points, points, self.r_cut + 0.5*self.diameter, nlist,
            exclude_ii=False)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        cdef locality.NeighborList * nlist_ptr = nlist_.get_ptr()

        with nogil:
            self.thisptr.compute(
                l_box, nlist_ptr,
                <vec3[float]*> l_ref_points.data,
                n_ref,
                <vec3[float]*> l_points.data,
                n_p)
        return self

    @property
    def density(self):
        return self.getDensity()

    def getDensity(self):
        """Get the density array for each particle.

        Returns:
            (:math:`N_{particles}`) :class:`numpy.ndarray`:
                Density array for each particle.
        """
        cdef float * density = self.thisptr.getDensity().get()
        cdef np.npy_intp nref[1]
        nref[0] = <np.npy_intp> self.thisptr.getNRef()
        cdef np.ndarray[np.float32_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(
                1, nref, np.NPY_FLOAT32, <void*> density)
        return result

    @property
    def num_neighbors(self):
        return self.getNumNeighbors()

    def getNumNeighbors(self):
        """Return the number of neighbors for each particle.

        Returns:
            (:math:`N_{particles}`) :class:`numpy.ndarray`:
                Number of neighbors for each particle.
        """
        cdef float * neighbors = self.thisptr.getNumNeighbors().get()
        cdef np.npy_intp nref[1]
        nref[0] = <np.npy_intp> self.thisptr.getNRef()
        cdef np.ndarray[np.float32_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(
                1, nref, np.NPY_FLOAT32, <void*> neighbors)
        return result

cdef class RDF:
    """ Computes RDF for supplied data.

    The RDF (:math:`g \\left( r \\right)`) is computed and averaged for a given
    set of reference points in a sea of data points. Providing the same points
    calculates them against themselves. Computing the RDF results in an RDF
    array listing the value of the RDF at each given :math:`r`, listed in the
    :code:`r` array.

    The values of :math:`r` to compute the RDF are set by the values of
    :code:`rmin`, :code:`rmax`, :code:`dr` in the constructor. :code:`rmax`
    sets the maximum distance at which to calculate the
    :math:`g \\left( r \\right)`, :code:`rmin` sets the minimum distance at
    which to calculate the :math:`g \\left( r \\right)`, and :code:`dr`
    determines the step size for each bin.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    .. note::
        2D: :py:class:`freud.density.RDF` properly handles 2D boxes.
        The points must be passed in as :code:`[x, y, 0]`.
        Failing to set z=0 will lead to undefined behavior.

    Args:
        rmax (float):
            Maximum distance to calculate.
        dr (float):
            Distance between histogram bins.
        rmin (float):
            Minimum distance to calculate, default 0.

    Attributes:
        box (:py:class:`freud.box.Box`):
            Box used in the calculation.
        RDF ((:math:`N_{bins}`) :class:`numpy.ndarray`):
            Histogram of RDF values.
        R ((:math:`N_{bins}`, 3) :class:`numpy.ndarray`):
            The values of bin centers.
        n_r ((:math:`N_{bins}`, 3) :class:`numpy.ndarray`):
            Histogram of cumulative RDF values.

    .. versionchanged:: 0.7.0
       Added optional `rmin` argument.
    """
    cdef density.RDF * thisptr
    cdef rmax

    def __cinit__(self, float rmax, float dr, float rmin=0):
        if rmax <= 0:
            raise ValueError("rmax must be > 0")
        if rmax <= rmin:
            raise ValueError("rmax must be > rmin")
        if dr <= 0.0:
            raise ValueError("dr must be > 0")
        self.thisptr = new density.RDF(rmax, dr, rmin)
        self.rmax = rmax

    def __dealloc__(self):
        del self.thisptr

    @property
    def box(self):
        return self.getBox()

    def getBox(self):
        """Get the box used in the calculation.

        Returns:
          :class:`freud.box.Box`: freud Box.
        """
        return BoxFromCPP(self.thisptr.getBox())

    def accumulate(self, box, ref_points, points, nlist=None):
        """Calculates the RDF and adds to the current RDF histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points to calculate the local density.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the bonding.
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList to use to find bonds (Default value = None).
        """
        box = freud.common.convert_box(box)
        ref_points = freud.common.convert_array(
            ref_points, 2, dtype=np.float32, contiguous=True,
            array_name="ref_points")
        points = freud.common.convert_array(
            points, 2, dtype=np.float32, contiguous=True, array_name="points")
        if ref_points.shape[1] != 3 or points.shape[1] != 3:
            raise ValueError("The 2nd dimension must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=2] l_ref_points = ref_points
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int n_ref = <unsigned int> ref_points.shape[0]
        cdef unsigned int n_p = <unsigned int> points.shape[0]

        cdef _box.Box l_box = _box.Box(
            box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
            box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())

        defaulted_nlist = make_default_nlist(
            box, ref_points, points, self.rmax, nlist)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        cdef locality.NeighborList * nlist_ptr = nlist_.get_ptr()

        with nogil:
            self.thisptr.accumulate(
                l_box, nlist_ptr,
                <vec3[float]*> l_ref_points.data,
                n_ref,
                <vec3[float]*> l_points.data,
                n_p)
        return self

    def compute(self, box, ref_points, points, nlist=None):
        """Calculates the RDF for the specified points. Will overwrite the current
        histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points to calculate the local density.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the bonding.
            nlist (:class:`freud.locality.NeighborList`):
                NeighborList to use to find bonds (Default value = None)
        """
        self.thisptr.reset()
        self.accumulate(box, ref_points, points, nlist)
        return self

    def resetRDF(self):
        """Resets the values of RDF in memory."""
        self.thisptr.reset()

    def reduceRDF(self):
        """Reduces the histogram in the values over N processors to a single
        histogram. This is called automatically by
        :py:meth:`freud.density.RDF.getRDF()`,
        :py:meth:`freud.density.RDF.getNr()`.
        """
        self.thisptr.reduceRDF()

    @property
    def RDF(self):
        return self.getRDF()

    def getRDF(self):
        """Histogram of RDF values.

        Returns:
            (:math:`N_{bins}`, 3) :class:`numpy.ndarray`:
                Histogram of RDF values.
        """
        cdef float * rdf = self.thisptr.getRDF().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.thisptr.getNBins()
        cdef np.ndarray[np.float32_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(
                1, nbins, np.NPY_FLOAT32, <void*> rdf)
        return result

    @property
    def R(self):
        return self.getR()

    def getR(self):
        """Get values of the histogram bin centers.

        Returns:
            (:math:`N_{bins}`, 3) :class:`numpy.ndarray`:
                Values of the histogram bin centers.
        """
        cdef float * r = self.thisptr.getR().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.thisptr.getNBins()
        cdef np.ndarray[np.float32_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(
                1, nbins, np.NPY_FLOAT32, <void*> r)
        return result

    @property
    def n_r(self):
        return self.getNr()

    def getNr(self):
        """Get the histogram of cumulative RDF values.

        Returns:
            (:math:`N_{bins}`, 3) :class:`numpy.ndarray`:
                Histogram of cumulative RDF values.
        """
        cdef float * Nr = self.thisptr.getNr().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.thisptr.getNBins()
        cdef np.ndarray[np.float32_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(
                1, nbins, np.NPY_FLOAT32, <void*> Nr)
        return result
