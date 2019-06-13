# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

R"""
The :class:`freud.density` module contains various classes relating to the
density of the system. These functions allow evaluation of particle
distributions with respect to other particles.
"""

import freud.common
import freud.locality
import warnings
import numpy as np

from freud.util._VectorMath cimport vec3
from cython.operator cimport dereference

cimport freud._density
cimport freud.box, freud.locality
cimport numpy as np

# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef class FloatCF:
    R"""Computes the real pairwise correlation function.

    The correlation function is given by
    :math:`C(r) = \left\langle s_1(0) \cdot s_2(r) \right\rangle` between
    two sets of points :math:`p_1` (:code:`ref_points`) and :math:`p_2`
    (:code:`points`) with associated values :math:`s_1` (:code:`ref_values`)
    and :math:`s_2` (:code:`values`). Computing the correlation function
    results in an array of the expected (average) product of all values at a
    given radial distance :math:`r`.

    The values of :math:`r` where the correlation function is computed are
    controlled by the :code:`rmax` and :code:`dr` parameters to the
    constructor. :code:`rmax` determines the maximum distance at which to
    compute the correlation function and :code:`dr` is the step size for each
    bin.

    .. note::
        **2D:** :class:`freud.density.FloatCF` properly handles 2D boxes.
        The points must be passed in as :code:`[x, y, 0]`.
        Failing to set z=0 will lead to undefined behavior.

    .. note::
        **Self-correlation:** It is often the case that we wish to compute the
        correlation function of a set of points with itself. If :code:`points`
        is the same as :code:`ref_points`, not provided, or :code:`None`, we
        omit accumulating the self-correlation value in the first bin.

    .. moduleauthor:: Matthew Spellings <mspells@umich.edu>

    Args:
        rmax (float):
            Maximum pointwise distance to include in the calculation.
        dr (float):
            Bin size.

    Attributes:
        RDF ((:math:`N_{bins}`) :class:`numpy.ndarray`):
            Expected (average) product of all values whose radial distance
            falls within a given distance bin.
        box (:class:`freud.box.Box`):
            The box used in the calculation.
        counts ((:math:`N_{bins}`) :class:`numpy.ndarray`):
            The number of points in each histogram bin.
        R ((:math:`N_{bins}`) :class:`numpy.ndarray`):
            The centers of each bin.
    """
    cdef freud._density.CorrelationFunction[double] * thisptr
    cdef rmax
    cdef dr

    def __cinit__(self, float rmax, float dr):
        if dr <= 0.0:
            raise ValueError("dr must be > 0")
        self.thisptr = new freud._density.CorrelationFunction[double](rmax, dr)
        self.rmax = rmax
        self.dr = dr

    def __dealloc__(self):
        del self.thisptr

    def accumulate(self, box, ref_points, ref_values, points=None, values=None,
                   nlist=None):
        R"""Calculates the correlation function and adds to the current
        histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{ref\_points}`, 3) :class:`numpy.ndarray`):
                Reference points used to calculate the correlation function.
            ref_values ((:math:`N_{ref\_points}`) :class:`numpy.ndarray`):
                Real values used to calculate the correlation function.
            points ((:math:`N_{points}`, 3) :class:`numpy.ndarray`, optional):
                Points used to calculate the correlation function.
                Uses :code:`ref_points` if not provided or :code:`None`.
            values ((:math:`N_{points}`) :class:`numpy.ndarray`, optional):
                Real values used to calculate the correlation function.
                Uses :code:`ref_values` if not provided or :code:`None`.
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList to use to find bonds (Default value =
                :code:`None`).
        """
        cdef freud.box.Box b = freud.common.convert_box(box)
        if points is None:
            points = ref_points
        if values is None:
            values = ref_values
        ref_points = freud.common.convert_array(ref_points, shape=(None, 3))
        points = freud.common.convert_array(points, shape=(None, 3))
        ref_values = freud.common.convert_array(
            ref_values, shape=(ref_points.shape[0], ), dtype=np.float64)
        values = freud.common.convert_array(
            values, shape=(points.shape[0], ), dtype=np.float64)
        cdef const float[:, ::1] l_ref_points = ref_points
        cdef const float[:, ::1] l_points
        if ref_points is points:
            l_points = l_ref_points
        else:
            l_points = points
        cdef const double[::1] l_ref_values = ref_values
        cdef const double[::1] l_values
        if values is ref_values:
            l_values = l_ref_values
        else:
            l_values = values

        defaulted_nlist = freud.locality.make_default_nlist(
            b, ref_points, points, self.rmax, nlist, None)
        cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]

        cdef unsigned int n_ref = l_ref_points.shape[0]
        cdef unsigned int n_p = l_points.shape[0]
        with nogil:
            self.thisptr.accumulate(
                dereference(b.thisptr), nlist_.get_ptr(),
                <vec3[float]*> &l_ref_points[0, 0],
                <double*> &l_ref_values[0], n_ref,
                <vec3[float]*> &l_points[0, 0],
                <double*> &l_values[0],
                n_p)
        return self

    @property
    def RDF(self):
        cdef unsigned int n_bins = self.thisptr.getNBins()
        cdef const double[::1] RDF = \
            <double[:n_bins]> self.thisptr.getRDF().get()
        return np.asarray(RDF)

    @property
    def box(self):
        return freud.box.BoxFromCPP(self.thisptr.getBox())

    def reset(self):
        R"""Resets the values of the correlation function histogram in
        memory.
        """
        self.thisptr.reset()

    def compute(self, box, ref_points, ref_values, points=None, values=None,
                nlist=None):
        R"""Calculates the correlation function for the given points. Will
        overwrite the current histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{ref\_points}`, 3) :class:`numpy.ndarray`):
                Reference points used to calculate the correlation function.
            ref_values ((:math:`N_{ref\_points}`) :class:`numpy.ndarray`):
                Real values used to calculate the correlation function.
            points ((:math:`N_{points}`, 3) :class:`numpy.ndarray`, optional):
                Points used to calculate the correlation function.
                Uses :code:`ref_points` if not provided or :code:`None`.
            values ((:math:`N_{points}`) :class:`numpy.ndarray`, optional):
                Real values used to calculate the correlation function.
                Uses :code:`ref_values` if not provided or :code:`None`.
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList to use to find bonds (Default value =
                :code:`None`).
        """
        self.reset()
        self.accumulate(box, ref_points, ref_values, points, values, nlist)
        return self

    @property
    def counts(self):
        cdef unsigned int n_bins = self.thisptr.getNBins()
        cdef const unsigned int[::1] counts = \
            <unsigned int[:n_bins]> self.thisptr.getCounts().get()
        return np.asarray(counts, dtype=np.uint32)

    @property
    def R(self):
        cdef unsigned int n_bins = self.thisptr.getNBins()
        cdef const float[::1] R = \
            <float[:n_bins]> self.thisptr.getR().get()
        return np.asarray(R)

    def __repr__(self):
        return ("freud.density.{cls}(rmax={rmax}, dr={dr})").format(
            cls=type(self).__name__, rmax=self.rmax, dr=self.dr)

    def __str__(self):
        return repr(self)


cdef class ComplexCF:
    R"""Computes the complex pairwise correlation function.

    The correlation function is given by
    :math:`C(r) = \left\langle s_1(0) \cdot s_2(r) \right\rangle` between
    two sets of points :math:`p_1` (:code:`ref_points`) and :math:`p_2`
    (:code:`points`) with associated values :math:`s_1` (:code:`ref_values`)
    and :math:`s_2` (:code:`values`). Computing the correlation function
    results in an array of the expected (average) product of all values at a
    given radial distance :math:`r`.

    The values of :math:`r` where the correlation function is computed are
    controlled by the :code:`rmax` and :code:`dr` parameters to the
    constructor. :code:`rmax` determines the maximum distance at which to
    compute the correlation function and :code:`dr` is the step size for each
    bin.

    .. note::
        **2D:** :class:`freud.density.ComplexCF` properly handles 2D boxes.
        The points must be passed in as :code:`[x, y, 0]`.
        Failing to set z=0 will lead to undefined behavior.

    .. note::
        **Self-correlation:** It is often the case that we wish to compute the
        correlation function of a set of points with itself. If :code:`points`
        is the same as :code:`ref_points`, not provided, or :code:`None`, we
        omit accumulating the self-correlation value in the first bin.

    .. moduleauthor:: Matthew Spellings <mspells@umich.edu>

    Args:
        rmax (float):
            Maximum pointwise distance to include in the calculation.
        dr (float):
            Bin size.

    Attributes:
        RDF ((:math:`N_{bins}`) :class:`numpy.ndarray`):
            Expected (average) product of all values at a given radial
            distance.
        box (:class:`freud.box.Box`):
            Box used in the calculation.
        counts ((:math:`N_{bins}`) :class:`numpy.ndarray`):
            The number of points in each histogram bin.
        R ((:math:`N_{bins}`) :class:`numpy.ndarray`):
            The centers of each bin.
    """
    cdef freud._density.CorrelationFunction[np.complex128_t] * thisptr
    cdef rmax
    cdef dr

    def __cinit__(self, float rmax, float dr):
        if dr <= 0.0:
            raise ValueError("dr must be > 0")
        self.thisptr = new freud._density.CorrelationFunction[np.complex128_t](
            rmax, dr)
        self.rmax = rmax
        self.dr = dr

    def __dealloc__(self):
        del self.thisptr

    def accumulate(self, box, ref_points, ref_values, points=None, values=None,
                   nlist=None):
        R"""Calculates the correlation function and adds to the current
        histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{ref\_points}`, 3) :class:`numpy.ndarray`):
                Reference points used to calculate the correlation function.
            ref_values ((:math:`N_{ref\_points}`) :class:`numpy.ndarray`):
                Complex values used to calculate the correlation function.
            points ((:math:`N_{points}`, 3) :class:`numpy.ndarray`, optional):
                Points used to calculate the correlation function.
                Uses :code:`ref_points` if not provided or :code:`None`.
            values ((:math:`N_{points}`) :class:`numpy.ndarray`, optional):
                Complex values used to calculate the correlation function.
                Uses :code:`ref_values` if not provided or :code:`None`.
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList to use to find bonds (Default value =
                :code:`None`).
        """
        cdef freud.box.Box b = freud.common.convert_box(box)
        if points is None:
            points = ref_points
        if values is None:
            values = ref_values
        ref_points = freud.common.convert_array(ref_points, shape=(None, 3))
        points = freud.common.convert_array(points, shape=(None, 3))
        ref_values = freud.common.convert_array(
            ref_values, shape=(ref_points.shape[0], ), dtype=np.complex128)
        values = freud.common.convert_array(
            values, shape=(points.shape[0], ), dtype=np.complex128)
        cdef const float[:, ::1] l_ref_points = ref_points
        cdef const float[:, ::1] l_points
        if ref_points is points:
            l_points = l_ref_points
        else:
            l_points = points
        cdef np.complex128_t[::1] l_ref_values = ref_values
        cdef np.complex128_t[::1] l_values
        if values is ref_values:
            l_values = l_ref_values
        else:
            l_values = values

        defaulted_nlist = freud.locality.make_default_nlist(
            b, ref_points, points, self.rmax, nlist, None)
        cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]

        cdef unsigned int n_ref = l_ref_points.shape[0]
        cdef unsigned int n_p = l_points.shape[0]
        with nogil:
            self.thisptr.accumulate(
                dereference(b.thisptr), nlist_.get_ptr(),
                <vec3[float]*> &l_ref_points[0, 0],
                <np.complex128_t*> &l_ref_values[0],
                n_ref,
                <vec3[float]*> &l_points[0, 0],
                <np.complex128_t*> &l_values[0],
                n_p)
        return self

    @property
    def RDF(self):
        cdef unsigned int n_bins = self.thisptr.getNBins()
        cdef np.complex128_t[::1] RDF = \
            <np.complex128_t[:n_bins]> self.thisptr.getRDF().get()
        return np.asarray(RDF)

    @property
    def box(self):
        return freud.box.BoxFromCPP(self.thisptr.getBox())

    def reset(self):
        R"""Resets the values of the correlation function histogram in
        memory.
        """
        self.thisptr.reset()

    def compute(self, box, ref_points, ref_values, points=None, values=None,
                nlist=None):
        R"""Calculates the correlation function for the given points. Will
        overwrite the current histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{ref\_points}`, 3) :class:`numpy.ndarray`):
                Reference points used to calculate the correlation function.
            ref_values ((:math:`N_{ref\_points}`) :class:`numpy.ndarray`):
                Complex values used to calculate the correlation function.
            points ((:math:`N_{points}`, 3) :class:`numpy.ndarray`, optional):
                Points used to calculate the correlation function.
                Uses :code:`ref_points` if not provided or :code:`None`.
            values ((:math:`N_{points}`) :class:`numpy.ndarray`, optional):
                Complex values used to calculate the correlation function.
                Uses :code:`ref_values` if not provided or :code:`None`.
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList to use to find bonds (Default value =
                :code:`None`).
        """
        self.reset()
        self.accumulate(box, ref_points, ref_values, points, values, nlist)
        return self

    @property
    def counts(self):
        cdef unsigned int n_bins = self.thisptr.getNBins()
        cdef const unsigned int[::1] counts = \
            <unsigned int[:n_bins]> self.thisptr.getCounts().get()
        return np.asarray(counts, dtype=np.uint32)

    @property
    def R(self):
        cdef unsigned int n_bins = self.thisptr.getNBins()
        cdef const float[::1] R = \
            <float[:n_bins]> self.thisptr.getR().get()
        return np.asarray(R)

    def __repr__(self):
        return ("freud.density.{cls}(rmax={rmax}, dr={dr})").format(
            cls=type(self).__name__, rmax=self.rmax, dr=self.dr)

    def __str__(self):
        return repr(self)


cdef class GaussianDensity:
    R"""Computes the density of a system on a grid.

    Replaces particle positions with a Gaussian blur and calculates the
    contribution from each to the proscribed grid based upon the distance of
    the grid cell from the center of the Gaussian. The resulting data is a
    regular grid of particle densities that can be used in standard algorithms
    requiring evenly spaced point, such as Fast Fourier Transforms. The
    dimensions of the image (grid) are set in the constructor, and can either
    be set equally for all dimensions or for each dimension independently.

    - Constructor Calls:

        Initialize with all dimensions identical::

            freud.density.GaussianDensity(width, r_cut, sigma)

        Initialize with each dimension specified::

            freud.density.GaussianDensity(width_x, width_y, width_z, r_cut, sigma)

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
        box (:class:`freud.box.Box`):
            Box used in the calculation.
        gaussian_density ((:math:`w_x`, :math:`w_y`, :math:`w_z`) :class:`numpy.ndarray`):
            The image grid with the Gaussian density.
        counts ((:math:`N_{bins}`) :class:`numpy.ndarray`):
            The number of points in each histogram bin.
        R ((:math:`N_{bins}`) :class:`numpy.ndarray`):
            The centers of each bin.
    """  # noqa: E501
    cdef freud._density.GaussianDensity * thisptr
    cdef arglist

    def __cinit__(self, *args):
        if len(args) == 3:
            self.thisptr = new freud._density.GaussianDensity(
                args[0], args[1], args[2])
            self.arglist = [args[0], args[1], args[2]]
        elif len(args) == 5:
            self.thisptr = new freud._density.GaussianDensity(
                args[0], args[1], args[2], args[3], args[4])
            self.arglist = [args[0], args[1], args[2], args[3], args[4]]
        else:
            raise TypeError('GaussianDensity takes exactly 3 or 5 arguments')

    @property
    def box(self):
        return freud.box.BoxFromCPP(self.thisptr.getBox())

    def compute(self, box, points):
        R"""Calculates the Gaussian blur for the specified points. Does not
        accumulate (will overwrite current image).

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            points ((:math:`N_{points}`, 3) :class:`numpy.ndarray`):
                Points to calculate the local density.
        """
        cdef freud.box.Box b = freud.common.convert_box(box)
        points = freud.common.convert_array(points, shape=(None, 3))
        cdef const float[:, ::1] l_points = points
        cdef unsigned int n_p = points.shape[0]
        with nogil:
            self.thisptr.compute(dereference(b.thisptr),
                                 <vec3[float]*> &l_points[0, 0], n_p)
        return self

    @property
    def gaussian_density(self):
        cdef unsigned int width_x = self.thisptr.getWidthX()
        cdef unsigned int width_y = self.thisptr.getWidthY()
        cdef unsigned int width_z = self.thisptr.getWidthZ()
        cdef unsigned int array_size = width_x * width_y
        cdef freud.box.Box box = self.box
        if not box.is2D():
            array_size *= width_z
        cdef const float[::1] density = \
            <float[:array_size]> self.thisptr.getDensity().get()
        if box.is2D():
            array_shape = (width_y, width_x)
        else:
            array_shape = (width_z, width_y, width_x)
        return np.reshape(np.asarray(density), array_shape)

    def __repr__(self):
        if len(self.arglist) == 3:
            return ("freud.density.{cls}({width}, " +
                    "{r_cut}, {sigma})").format(cls=type(self).__name__,
                                                width=self.arglist[0],
                                                r_cut=self.arglist[1],
                                                sigma=self.arglist[2])
        elif len(self.arglist) == 5:
            return ("freud.density.{cls}({width_x}, {width_y}, {width_z}, " +
                    "{r_cut}, {sigma})").format(cls=type(self).__name__,
                                                width_x=self.arglist[0],
                                                width_y=self.arglist[1],
                                                width_z=self.arglist[2],
                                                r_cut=self.arglist[3],
                                                sigma=self.arglist[4])
        else:
            raise TypeError('GaussianDensity takes exactly 3 or 5 arguments')

    def __str__(self):
        return repr(self)


cdef class LocalDensity:
    R"""Computes the local density around a particle.

    The density of the local environment is computed and averaged for a given
    set of reference points in a sea of data points. Providing the same points
    calculates them against themselves. Computing the local density results in
    an array listing the value of the local density around each reference
    point. Also available is the number of neighbors for each reference point,
    giving the user the ability to count the number of particles in that
    region.

    The values to compute the local density are set in the constructor.
    :code:`r_cut` sets the maximum distance at which data points are included
    relative to a given reference point. :code:`volume` is the volume of a
    single data points, and :code:`diameter` is the diameter of the
    circumsphere of an individual data point. Note that the volume and diameter
    do not affect the reference point; whether or not data points are counted
    as neighbors of a given reference point is entirely determined by the
    distance between reference point and data point center relative to
    :code:`r_cut` and the :code:`diameter` of the data point.

    In order to provide sufficiently smooth data, data points can be
    fractionally counted towards the density.  Rather than perform
    compute-intensive area (volume) overlap calculations to
    determine the exact amount of overlap area (volume), the LocalDensity class
    performs a simple linear interpolation relative to the centers of the data
    points.  Specifically, a point is counted as one neighbor of a given
    reference point if it is entirely contained within the :code:`r_cut`, half
    of a neighbor if the distance to its center is exactly :code:`r_cut`, and
    zero if its center is a distance greater than or equal to :code:`r_cut +
    diameter` from the reference point's center. Graphically, this looks like:

    .. image:: images/density.png

    .. note::
        **2D:** :class:`freud.density.LocalDensity` properly handles 2D
        boxes. The points must be passed in as :code:`[x, y, 0]`. Failing to
        set z=0 will lead to undefined behavior.

    .. moduleauthor:: Joshua Anderson <joaander@umich.edu>

    Args:
        r_cut (float):
            Maximum distance over which to calculate the density.
        volume (float):
            Volume of a single particle.
        diameter (float):
            Diameter of particle circumsphere.

    Attributes:
        box (:class:`freud.box.Box`):
            Box used in the calculation.
        density ((:math:`N_{ref\_points}`) :class:`numpy.ndarray`):
            Density of points per ref_point.
        num_neighbors ((:math:`N_{ref\_points}`) :class:`numpy.ndarray`):
            Number of neighbor points for each ref_point.
    """
    cdef freud._density.LocalDensity * thisptr
    cdef r_cut
    cdef diameter
    cdef volume

    def __cinit__(self, float r_cut, float volume, float diameter):
        self.thisptr = new freud._density.LocalDensity(r_cut, volume, diameter)
        self.r_cut = r_cut
        self.diameter = diameter
        self.volume = volume

    @property
    def box(self):
        return freud.box.BoxFromCPP(self.thisptr.getBox())

    def compute(self, box, ref_points, points=None, nlist=None):
        R"""Calculates the local density for the specified points. Does not
        accumulate (will overwrite current data).

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{ref\_points}`, 3) :class:`numpy.ndarray`):
                Reference points to calculate the local density.
            points ((:math:`N_{points}`, 3) :class:`numpy.ndarray`, optional):
                Points to calculate the local density. Uses :code:`ref_points`
                if not provided or :code:`None`.
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList to use to find bonds (Default value =
                :code:`None`).
        """
        cdef freud.box.Box b = freud.common.convert_box(box)
        if points is None:
            points = ref_points
        ref_points = freud.common.convert_array(ref_points, shape=(None, 3))
        points = freud.common.convert_array(points, shape=(None, 3))
        cdef const float[:, ::1] l_ref_points = ref_points
        cdef const float[:, ::1] l_points = points
        cdef unsigned int n_ref = l_ref_points.shape[0]
        cdef unsigned int n_p = l_points.shape[0]

        # local density of each particle includes itself (cutoff
        # distance is r_cut + diam/2 because of smoothing)
        defaulted_nlist = freud.locality.make_default_nlist(
            b, ref_points, points, self.r_cut + 0.5*self.diameter, nlist,
            exclude_ii=False)
        cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]

        with nogil:
            self.thisptr.compute(
                dereference(b.thisptr), nlist_.get_ptr(),
                <vec3[float]*> &l_ref_points[0, 0],
                n_ref,
                <vec3[float]*> &l_points[0, 0],
                n_p)
        return self

    @property
    def density(self):
        cdef unsigned int n_ref = self.thisptr.getNRef()
        cdef const float[::1] density = \
            <float[:n_ref]> self.thisptr.getDensity().get()
        return np.asarray(density)

    @property
    def num_neighbors(self):
        cdef unsigned int n_ref = self.thisptr.getNRef()
        cdef const float[::1] num_neighbors = \
            <float[:n_ref]> self.thisptr.getNumNeighbors().get()
        return np.asarray(num_neighbors)

    def __repr__(self):
        return ("freud.density.{cls}(r_cut={r_cut}, volume={volume}, " +
                "diameter={diameter})").format(cls=type(self).__name__,
                                               r_cut=self.r_cut,
                                               volume=self.volume,
                                               diameter=self.diameter)

    def __str__(self):
        return repr(self)


cdef class RDF:
    R"""Computes RDF for supplied data.

    The RDF (:math:`g \left( r \right)`) is computed and averaged for a given
    set of reference points in a sea of data points. Providing the same points
    calculates them against themselves. Computing the RDF results in an RDF
    array listing the value of the RDF at each given :math:`r`, listed in the
    :code:`R` array.

    The values of :math:`r` to compute the RDF are set by the values of
    :code:`rmin`, :code:`rmax`, :code:`dr` in the constructor. :code:`rmax`
    sets the maximum distance at which to calculate the
    :math:`g \left( r \right)`, :code:`rmin` sets the minimum distance at
    which to calculate the :math:`g \left( r \right)`, and :code:`dr`
    determines the step size for each bin.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    .. note::
        **2D:** :class:`freud.density.RDF` properly handles 2D boxes.
        The points must be passed in as :code:`[x, y, 0]`.
        Failing to set z=0 will lead to undefined behavior.

    Args:
        rmax (float):
            Maximum interparticle distance to include in the calculation.
        dr (float):
            Distance between histogram bins.
        rmin (float, optional):
            Minimum interparticle distance to include in the calculation
            (Default value = 0).

    Attributes:
        box (:class:`freud.box.Box`):
            Box used in the calculation.
        RDF ((:math:`N_{bins}`,) :class:`numpy.ndarray`):
            Histogram of RDF values.
        R ((:math:`N_{bins}`) :class:`numpy.ndarray`):
            The centers of each bin.
        n_r ((:math:`N_{bins}`,) :class:`numpy.ndarray`):
            Histogram of cumulative RDF values (*i.e.* the integrated RDF).

    .. versionchanged:: 0.7.0
       Added optional `rmin` argument.
    """
    cdef freud._density.RDF * thisptr
    cdef rmax
    cdef dr
    cdef rmin

    def __cinit__(self, float rmax, float dr, float rmin=0):
        if rmax <= 0:
            raise ValueError("rmax must be > 0")
        if rmax <= rmin:
            raise ValueError("rmax must be > rmin")
        if dr <= 0.0:
            raise ValueError("dr must be > 0")
        self.thisptr = new freud._density.RDF(rmax, dr, rmin)
        self.rmax = rmax
        self.dr = dr
        self.rmin = rmin

    def __dealloc__(self):
        del self.thisptr

    @property
    def box(self):
        return freud.box.BoxFromCPP(self.thisptr.getBox())

    def accumulate(self, box, ref_points, points=None, nlist=None):
        R"""Calculates the RDF and adds to the current RDF histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{ref\_points}`, 3) :class:`numpy.ndarray`):
                Reference points used to calculate the RDF.
            points ((:math:`N_{points}`, 3) :class:`numpy.ndarray`, optional):
                Points used to calculate the RDF. Uses :code:`ref_points` if
                not provided or :code:`None`.
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList to use to find bonds (Default value =
                :code:`None`).
        """
        cdef freud.box.Box b = freud.common.convert_box(box)
        if points is None:
            points = ref_points
        ref_points = freud.common.convert_array(ref_points, shape=(None, 3))
        points = freud.common.convert_array(points, shape=(None, 3))
        cdef const float[:, ::1] l_ref_points = ref_points
        cdef const float[:, ::1] l_points = points
        cdef unsigned int n_ref = l_ref_points.shape[0]
        cdef unsigned int n_p = l_points.shape[0]

        defaulted_nlist = freud.locality.make_default_nlist(
            b, ref_points, points, self.rmax, nlist)
        cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]

        with nogil:
            self.thisptr.accumulate(
                dereference(b.thisptr), nlist_.get_ptr(),
                <vec3[float]*> &l_ref_points[0, 0],
                n_ref,
                <vec3[float]*> &l_points[0, 0],
                n_p)
        return self

    def compute(self, box, ref_points, points=None, nlist=None):
        R"""Calculates the RDF for the specified points. Will overwrite the current
        histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{ref\_points}`, 3) :class:`numpy.ndarray`):
                Reference points used to calculate the RDF.
            points ((:math:`N_{points}`, 3) :class:`numpy.ndarray`, optional):
                Points used to calculate the RDF. Uses :code:`ref_points` if
                not provided or :code:`None`.
            nlist (:class:`freud.locality.NeighborList`):
                NeighborList to use to find bonds (Default value =
                :code:`None`).
        """
        self.reset()
        self.accumulate(box, ref_points, points, nlist)
        return self

    def reset(self):
        R"""Resets the values of RDF in memory."""
        self.thisptr.reset()

    @property
    def RDF(self):
        cdef unsigned int n_bins = self.thisptr.getNBins()
        cdef const float[::1] RDF = \
            <float[:n_bins]> self.thisptr.getRDF().get()
        return np.asarray(RDF)

    @property
    def R(self):
        cdef unsigned int n_bins = self.thisptr.getNBins()
        cdef const float[::1] R = \
            <float[:n_bins]> self.thisptr.getR().get()
        return np.asarray(R)

    @property
    def n_r(self):
        cdef unsigned int n_bins = self.thisptr.getNBins()
        cdef const float[::1] n_r = <float[:n_bins]> self.thisptr.getNr().get()
        return np.asarray(n_r)

    def __repr__(self):
        return ("freud.density.{cls}(rmax={rmax}, dr={dr}, " +
                "rmin={rmin})").format(cls=type(self).__name__,
                                       rmax=self.rmax,
                                       dr=self.dr,
                                       rmin=self.rmin)

    def __str__(self):
        return repr(self)
