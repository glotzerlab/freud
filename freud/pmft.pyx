# Copyright (c) 2010-2020 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

R"""
The :class:`freud.pmft` module allows for the calculation of the Potential of
Mean Force and Torque (PMFT) :cite:`vanAnders:2014aa,van_Anders_2013` in a
number of different coordinate systems. The shape of the arrays computed by
this module depend on the coordinate system used, with space discretized into a
set of bins created by the PMFT object's constructor. Each query point's
neighboring points are assigned to bins, determined by the relative positions
and/or orientations of the particles. Next, the pair correlation function
(PCF) is computed by normalizing the binned histogram, by dividing out the
number of accumulated frames, bin sizes (the Jacobian), and query point
number density. The PMFT is then defined as the negative logarithm of the PCF.
For further descriptions of the numerical methods used to compute the PMFT,
refer to the supplementary information of :cite:`vanAnders:2014aa`.

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

.. note::
    For any bins where the histogram is zero (i.e. no observations were made
    with that relative position/orientation of particles), the PMFT will return
    :code:`nan`.
"""

import numpy as np
import rowan

import freud.locality

cimport numpy as np
from cython.operator cimport dereference

cimport freud._pmft
cimport freud.locality
from freud.locality cimport _SpatialHistogram
from freud.util cimport _Compute, quat, vec3

# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()


def _quat_to_z_angle(orientations, num_points):
    """If orientations are quaternions, convert them to angles.

    For consistency with the boxes and points in freud, we require that
    the orientations represent rotations about the z-axis. If last
    dimension is of length 4, it's a quaternion, unless we have exactly
    4 points, in which case it could be 4 angles. In that case, we also
    have to check that orientations are of shape (4, 4)
    """
    # Either we have a 1D array of length 4 (and we don't have exactly 4
    # points), or we have a 2D array with the second dimension having length 4.
    is_quat = (
        (len(orientations.shape) == 1 and orientations.shape[0] == 4 and
            num_points != 4) or
        (len(orientations.shape) == 2 and orientations.shape[1] == 4)
    )

    if is_quat:
        axes, orientations = rowan.to_axis_angle(orientations)
        if not (np.allclose(orientations, 0) or np.allclose(axes, [0, 0, 1])):
            raise ValueError("Orientations provided as quaternions "
                             "must represent rotations about the z-axis.")
    return orientations


def _gen_angle_array(orientations, shape):
    """Generates properly shaped, freud-compliant arrays of angles.

    This computation is specific to 2D calculations that require angles as
    orientations. It performs the conversion of quaternion inputs if needed and
    ensures that singleton arrays are treated correctly."""

    return freud.util._convert_array(np.atleast_1d(_quat_to_z_angle(
        np.asarray(orientations).squeeze(), shape[0])), shape=shape)


cdef class _PMFT(_SpatialHistogram):
    R"""Compute the PMFT :cite:`vanAnders:2014aa,van_Anders_2013` for a
    given set of points.

    This class provides an abstract interface for computing the PMFT.
    It must be specialized for a specific coordinate system; although in
    principle the PMFT is coordinate independent, the binning process must be
    performed in a particular coordinate system.
    """
    cdef freud._pmft.PMFT * pmftptr

    def __dealloc__(self):
        if type(self) is _PMFT:
            del self.pmftptr

    @_Compute._computed_property
    def pmft(self):
        """:class:`np.ndarray`: The discrete potential of mean force and
        torque."""
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore')
            result = -np.log(np.copy(self._pcf))
        return result

    @_Compute._computed_property
    def _pcf(self):
        """:class:`np.ndarray`: The discrete pair correlation function."""
        return freud.util.make_managed_numpy_array(
            &self.pmftptr.getPCF(),
            freud.util.arr_type_t.FLOAT)


cdef class PMFTR12(_PMFT):
    R"""Computes the PMFT :cite:`vanAnders:2014aa,van_Anders_2013` in a 2D
    system described by :math:`r`, :math:`\theta_1`, :math:`\theta_2`.

    .. note::
        **2D:** :class:`freud.pmft.PMFTR12` is only defined for 2D systems.
        The points must be passed in as :code:`[x, y, 0]`.

    Args:
        r_max (float):
            Maximum distance at which to compute the PMFT.
        bins (unsigned int or sequence of length 3):
            If an unsigned int, the number of bins in :math:`r`,
            :math:`\theta_1`, and :math:`\theta_2`. If a sequence of three
            integers, interpreted as :code:`(num_bins_r, num_bins_t1,
            num_bins_t2)`.
    """  # noqa: E501
    cdef freud._pmft.PMFTR12 * pmftr12ptr

    def __cinit__(self, r_max, bins):
        if type(self) is PMFTR12:
            try:
                n_r, n_t1, n_t2 = bins
            except TypeError:
                n_r = n_t1 = n_t2 = bins
            self.pmftr12ptr = self.pmftptr = self.histptr = \
                new freud._pmft.PMFTR12(r_max, n_r, n_t1, n_t2)
            self.r_max = r_max

    def __dealloc__(self):
        if type(self) is PMFTR12:
            del self.pmftr12ptr

    def compute(self, system, orientations, query_points=None,
                query_orientations=None, neighbors=None, reset=True):
        R"""Calculates the PMFT.

        Args:
            system:
                Any object that is a valid argument to
                :class:`freud.locality.NeighborQuery.from_system`.
            orientations ((:math:`N_{points}`, 4) or (:math:`N_{points}`,) :class:`numpy.ndarray`):
                Orientations associated with system points that are used to
                calculate bonds. If the array is one-dimensional, the values
                are treated as angles in radians corresponding to
                **counterclockwise** rotations about the z axis.
            query_points ((:math:`N_{query\_points}`, 3) :class:`numpy.ndarray`, optional):
                Query points used to calculate the PMFT. Uses the system's
                points if :code:`None` (Default value = :code:`None`).
            query_orientations ((:math:`N_{query\_points}`, 4) :class:`numpy.ndarray`, optional):
                Query orientations associated with query points that are used
                to calculate bonds. If the array is one-dimensional, the values
                are treated as angles in radians corresponding to
                **counterclockwise** rotations about the z axis. Uses
                :code:`orientations` if :code:`None`.  (Default value =
                :code:`None`).
            neighbors (:class:`freud.locality.NeighborList` or dict, optional):
                Either a :class:`NeighborList <freud.locality.NeighborList>` of
                neighbor pairs to use in the calculation, or a dictionary of
                `query arguments
                <https://freud.readthedocs.io/en/stable/topics/querying.html>`_
                (Default value: None).
            reset (bool):
                Whether to erase the previously computed values before adding
                the new computation; if False, will accumulate data (Default
                value: True).
        """  # noqa: E501
        if reset:
            self._reset()

        cdef:
            freud.locality.NeighborQuery nq
            freud.locality.NeighborList nlist
            freud.locality._QueryArgs qargs
            const float[:, ::1] l_query_points
            unsigned int num_query_points

        nq, nlist, qargs, l_query_points, num_query_points = \
            self._preprocess_arguments(
                system, query_points, neighbors)

        orientations = _gen_angle_array(
            orientations, shape=(nq.points.shape[0], ))
        if query_orientations is None:
            query_orientations = orientations
        else:
            query_orientations = _gen_angle_array(
                query_orientations, shape=(l_query_points.shape[0], ))
        cdef const float[::1] l_orientations = orientations
        cdef const float[::1] l_query_orientations = query_orientations

        self.pmftr12ptr.accumulate(nq.get_ptr(),
                                   <float*> &l_orientations[0],
                                   <vec3[float]*> &l_query_points[0, 0],
                                   <float*> &l_query_orientations[0],
                                   num_query_points, nlist.get_ptr(),
                                   dereference(qargs.thisptr))
        return self

    def __repr__(self):
        bounds = self.bounds
        return ("freud.pmft.{cls}(r_max={r_max}, bins=({bins}))").format(
            cls=type(self).__name__,
            r_max=self.r_max,
            bins=', '.join([str(b) for b in self.nbins]))


cdef class PMFTXYT(_PMFT):
    R"""Computes the PMFT :cite:`vanAnders:2014aa,van_Anders_2013` for
    systems described by coordinates :math:`x`, :math:`y`, :math:`\theta`.

    .. note::
        **2D:** :class:`freud.pmft.PMFTXYT` is only defined for 2D systems.
        The points must be passed in as :code:`[x, y, 0]`.

    Args:
        x_max (float):
            Maximum :math:`x` distance at which to compute the PMFT.
        y_max (float):
            Maximum :math:`y` distance at which to compute the PMFT.
        bins (unsigned int or sequence of length 3):
            If an unsigned int, the number of bins in :math:`x`, :math:`y`, and
            :math:`t`. If a sequence of three integers, interpreted as
            :code:`(num_bins_x, num_bins_y, num_bins_t)`.
    """  # noqa: E501
    cdef freud._pmft.PMFTXYT * pmftxytptr

    def __cinit__(self, x_max, y_max, bins):
        if type(self) is PMFTXYT:
            try:
                n_x, n_y, n_t = bins
            except TypeError:
                n_x = n_y = n_t = bins

            self.pmftxytptr = self.pmftptr = self.histptr = \
                new freud._pmft.PMFTXYT(x_max, y_max, n_x, n_y, n_t)
            self.r_max = np.sqrt(x_max**2 + y_max**2)

    def __dealloc__(self):
        if type(self) is PMFTXYT:
            del self.pmftxytptr

    def compute(self, system, orientations, query_points=None,
                query_orientations=None, neighbors=None, reset=True):
        R"""Calculates the PMFT.

        Args:
            system:
                Any object that is a valid argument to
                :class:`freud.locality.NeighborQuery.from_system`.
            orientations ((:math:`N_{points}`, 4) or (:math:`N_{points}`,) :class:`numpy.ndarray`):
                Orientations associated with system points that are used to
                calculate bonds. If the array is one-dimensional, the values
                are treated as angles in radians corresponding to
                **counterclockwise** rotations about the z axis.
            query_points ((:math:`N_{query\_points}`, 3) :class:`numpy.ndarray`, optional):
                Query points used to calculate the PMFT. Uses the system's
                points if :code:`None` (Default value = :code:`None`).
            query_orientations ((:math:`N_{query\_points}`, 4) :class:`numpy.ndarray`, optional):
                Query orientations associated with query points that are used
                to calculate bonds. If the array is one-dimensional, the values
                are treated as angles in radians corresponding to
                **counterclockwise** rotations about the z axis. Uses
                :code:`orientations` if :code:`None`.  (Default value =
                :code:`None`).
            neighbors (:class:`freud.locality.NeighborList` or dict, optional):
                Either a :class:`NeighborList <freud.locality.NeighborList>` of
                neighbor pairs to use in the calculation, or a dictionary of
                `query arguments
                <https://freud.readthedocs.io/en/stable/topics/querying.html>`_
                (Default value: None).
            reset (bool):
                Whether to erase the previously computed values before adding
                the new computation; if False, will accumulate data (Default
                value: True).
        """  # noqa: E501
        if reset:
            self._reset()

        cdef:
            freud.locality.NeighborQuery nq
            freud.locality.NeighborList nlist
            freud.locality._QueryArgs qargs
            const float[:, ::1] l_query_points
            unsigned int num_query_points

        nq, nlist, qargs, l_query_points, num_query_points = \
            self._preprocess_arguments(
                system, query_points, neighbors)

        orientations = _gen_angle_array(
            orientations, shape=(nq.points.shape[0], ))
        if query_orientations is None:
            query_orientations = orientations
        else:
            query_orientations = _gen_angle_array(
                query_orientations, shape=(l_query_points.shape[0], ))
        cdef const float[::1] l_orientations = orientations
        cdef const float[::1] l_query_orientations = query_orientations

        self.pmftxytptr.accumulate(nq.get_ptr(),
                                   <float*> &l_orientations[0],
                                   <vec3[float]*> &l_query_points[0, 0],
                                   <float*> &l_query_orientations[0],
                                   num_query_points, nlist.get_ptr(),
                                   dereference(qargs.thisptr))
        return self

    def __repr__(self):
        bounds = self.bounds
        return ("freud.pmft.{cls}(x_max={x_max}, y_max={y_max}, "
                "bins=({bins}))").format(cls=type(self).__name__,
                                         x_max=bounds[0][1],
                                         y_max=bounds[1][1],
                                         bins=', '.join(
                                             [str(b) for b in self.nbins]))


cdef class PMFTXY(_PMFT):
    R"""Computes the PMFT :cite:`vanAnders:2014aa,van_Anders_2013` in
    coordinates :math:`x`, :math:`y`.

    There are 3 degrees of translational and rotational freedom in 2
    dimensions, so this class implicitly integrates over the rotational degree
    of freedom of the second particle.

    .. note::
        **2D:** :class:`freud.pmft.PMFTXY` is only defined for 2D systems.
        The points must be passed in as :code:`[x, y, 0]`.

    Args:
        x_max (float):
            Maximum :math:`x` distance at which to compute the PMFT.
        y_max (float):
            Maximum :math:`y` distance at which to compute the PMFT.
        bins (unsigned int or sequence of length 2):
            If an unsigned int, the number of bins in :math:`x` and :math:`y`.
            If a sequence of two integers, interpreted as
            :code:`(num_bins_x, num_bins_y)`.
    """  # noqa: E501
    cdef freud._pmft.PMFTXY * pmftxyptr

    def __cinit__(self, x_max, y_max, bins):
        if type(self) is PMFTXY:
            try:
                n_x, n_y = bins
            except TypeError:
                n_x = n_y = bins

            self.pmftxyptr = self.pmftptr = self.histptr = \
                new freud._pmft.PMFTXY(x_max, y_max, n_x, n_y)
            self.r_max = np.sqrt(x_max**2 + y_max**2)

    def __dealloc__(self):
        if type(self) is PMFTXY:
            del self.pmftxyptr

    def compute(self, system, query_orientations, query_points=None,
                neighbors=None, reset=True):
        R"""Calculates the PMFT.

        .. note::
            The orientations of the system points are irrelevant for this
            calculation because that dimension is integrated out. The provided
            ``query_orientations`` are therefore always associated with
            ``query_points`` (which are equal to the system points if no
            ``query_points`` are explicitly provided).

        Args:
            system:
                Any object that is a valid argument to
                :class:`freud.locality.NeighborQuery.from_system`.
            query_orientations ((:math:`N_{query\_points}`, 4) or (:math:`N_{query\_points}`,) :class:`numpy.ndarray`):
                Query orientations associated with query points that are used
                to calculate bonds. If the array is one-dimensional, the values
                are treated as angles in radians corresponding to
                **counterclockwise** rotations about the z axis.
            query_points ((:math:`N_{query\_points}`, 3) :class:`numpy.ndarray`, optional):
                Query points used to calculate the PMFT. Uses the system's
                points if :code:`None` (Default value = :code:`None`).
            neighbors (:class:`freud.locality.NeighborList` or dict, optional):
                Either a :class:`NeighborList <freud.locality.NeighborList>` of
                neighbor pairs to use in the calculation, or a dictionary of
                `query arguments
                <https://freud.readthedocs.io/en/stable/topics/querying.html>`_
                (Default value: None).
            reset (bool):
                Whether to erase the previously computed values before adding
                the new computation; if False, will accumulate data (Default
                value: True).
        """  # noqa: E501
        if reset:
            self._reset()

        cdef:
            freud.locality.NeighborQuery nq
            freud.locality.NeighborList nlist
            freud.locality._QueryArgs qargs
            const float[:, ::1] l_query_points
            unsigned int num_query_points

        nq, nlist, qargs, l_query_points, num_query_points = \
            self._preprocess_arguments(
                system, query_points, neighbors)

        query_orientations = _gen_angle_array(
            query_orientations, shape=(num_query_points, ))
        cdef const float[::1] l_query_orientations = query_orientations

        self.pmftxyptr.accumulate(nq.get_ptr(),
                                  <float*> &l_query_orientations[0],
                                  <vec3[float]*> &l_query_points[0, 0],
                                  num_query_points, nlist.get_ptr(),
                                  dereference(qargs.thisptr))
        return self

    @_Compute._computed_property
    def bin_counts(self):
        """:class:`numpy.ndarray`: The bin counts in the histogram."""
        # Currently the parent function returns a 3D array that must be
        # squeezed due to the internal choices in the histogramming; this will
        # be fixed in future changes.
        return np.squeeze(super(PMFTXY, self).bin_counts)

    def __repr__(self):
        bounds = self.bounds
        return ("freud.pmft.{cls}(x_max={x_max}, y_max={y_max}, "
                "bins=({bins}))").format(cls=type(self).__name__,
                                         x_max=bounds[0][1],
                                         y_max=bounds[1][1],
                                         bins=', '.join(
                                             [str(b) for b in self.nbins]))

    def _repr_png_(self):
        try:
            import freud.plot
            return freud.plot._ax_to_bytes(self.plot())
        except (AttributeError, ImportError):
            return None

    def plot(self, ax=None):
        """Plot PMFTXY.

        Args:
            ax (:class:`matplotlib.axes.Axes`, optional): Axis to plot on. If
                :code:`None`, make a new figure and axis.
                (Default value = :code:`None`)

        Returns:
            (:class:`matplotlib.axes.Axes`): Axis with the plot.
        """
        import freud.plot
        return freud.plot.pmft_plot(self, ax)


cdef class PMFTXYZ(_PMFT):
    R"""Computes the PMFT :cite:`vanAnders:2014aa,van_Anders_2013` in
    coordinates :math:`x`, :math:`y`, :math:`z`.

    There are 6 degrees of translational and rotational freedom in 3
    dimensions, so this class is implicitly integrates out the orientational
    degrees of freedom in the system associated with the points. All
    calculations are done in the reference from of the query points.

    Args:
        x_max (float):
            Maximum :math:`x` distance at which to compute the PMFT.
        y_max (float):
            Maximum :math:`y` distance at which to compute the PMFT.
        z_max (float):
            Maximum :math:`z` distance at which to compute the PMFT.
        bins (unsigned int or sequence of length 3):
            If an unsigned int, the number of bins in :math:`x`, :math:`y`, and
            :math:`z`. If a sequence of three integers, interpreted as
            :code:`(num_bins_x, num_bins_y, num_bins_z)`.
        shiftvec (list):
            Vector pointing from ``[0, 0, 0]`` to the center of the PMFT.
    """  # noqa: E501
    cdef freud._pmft.PMFTXYZ * pmftxyzptr
    cdef shiftvec

    def __cinit__(self, x_max, y_max, z_max, bins,
                  shiftvec=[0, 0, 0]):
        cdef vec3[float] c_shiftvec

        try:
            n_x, n_y, n_z = bins
        except TypeError:
            n_x = n_y = n_z = bins

        if type(self) is PMFTXYZ:
            c_shiftvec = vec3[float](
                shiftvec[0], shiftvec[1], shiftvec[2])
            self.pmftxyzptr = self.pmftptr = self.histptr = \
                new freud._pmft.PMFTXYZ(
                    x_max, y_max, z_max, n_x, n_y, n_z, c_shiftvec)
            self.shiftvec = np.array(shiftvec, dtype=np.float32)
            self.r_max = np.sqrt(x_max**2 + y_max**2 + z_max**2)

    def __dealloc__(self):
        if type(self) is PMFTXYZ:
            del self.pmftxyzptr

    def compute(self, system, query_orientations, query_points=None,
                equiv_orientations=None, neighbors=None, reset=True):
        R"""Calculates the PMFT.

        .. note::
            The orientations of the system points are irrelevant for this
            calculation because that dimension is integrated out. The provided
            ``query_orientations`` are therefore always associated with
            ``query_points`` (which are equal to the system points if no
            ``query_points`` are explicitly provided.

        Args:
            system:
                Any object that is a valid argument to
                :class:`freud.locality.NeighborQuery.from_system`.
            query_orientations ((:math:`N_{points}`, 4) :class:`numpy.ndarray`):
                Query orientations associated with query points that are used
                to calculate bonds.
            query_points ((:math:`N_{query\_points}`, 3) :class:`numpy.ndarray`, optional):
                Query points used to calculate the PMFT. Uses the system's
                points if :code:`None` (Default value = :code:`None`).
            equiv_orientations ((:math:`N_{faces}`, 4) :class:`numpy.ndarray`, optional):
                Orientations to be treated as equivalent to account for
                symmetry of the points. For instance, if the
                :code:`query_points` are rectangular prisms with the long axis
                corresponding to the x-axis, then a point at :math:`(1, 0, 0)`
                and a point at :math:`(-1, 0, 0)` are symmetrically equivalent
                and can be counted towards both the positive and negative bins.
                If not supplied by user or :code:`None`, a unit quaternion will
                be used (Default value = :code:`None`).
            neighbors (:class:`freud.locality.NeighborList` or dict, optional):
                Either a :class:`NeighborList <freud.locality.NeighborList>` of
                neighbor pairs to use in the calculation, or a dictionary of
                `query arguments
                <https://freud.readthedocs.io/en/stable/topics/querying.html>`_
                (Default value: None).
            reset (bool):
                Whether to erase the previously computed values before adding
                the new computation; if False, will accumulate data (Default
                value: True).
        """  # noqa: E501
        if reset:
            self._reset()

        cdef:
            freud.locality.NeighborQuery nq
            freud.locality.NeighborList nlist
            freud.locality._QueryArgs qargs
            const float[:, ::1] l_query_points
            unsigned int num_query_points

        nq, nlist, qargs, l_query_points, num_query_points = \
            self._preprocess_arguments(
                system, query_points, neighbors)
        l_query_points = l_query_points - self.shiftvec.reshape(1, 3)

        query_orientations = freud.util._convert_array(
            np.atleast_1d(query_orientations), shape=(num_query_points, 4))

        cdef const float[:, ::1] l_query_orientations = query_orientations

        if equiv_orientations is None:
            equiv_orientations = np.array([[1, 0, 0, 0]], dtype=np.float32)
        else:
            equiv_orientations = freud.util._convert_array(
                equiv_orientations, shape=(None, 4))

        cdef const float[:, ::1] l_equiv_orientations = equiv_orientations
        cdef unsigned int num_equiv_orientations = \
            l_equiv_orientations.shape[0]
        self.pmftxyzptr.accumulate(
            nq.get_ptr(),
            <quat[float]*> &l_query_orientations[0, 0],
            <vec3[float]*> &l_query_points[0, 0],
            num_query_points,
            <quat[float]*> &l_equiv_orientations[0, 0],
            num_equiv_orientations, nlist.get_ptr(),
            dereference(qargs.thisptr))
        return self

    def __repr__(self):
        bounds = self.bounds
        return ("freud.pmft.{cls}(x_max={x_max}, y_max={y_max}, "
                "z_max={z_max}, bins=({bins}), "
                "shiftvec={shiftvec})").format(
                    cls=type(self).__name__,
                    x_max=bounds[0][1],
                    y_max=bounds[1][1],
                    z_max=bounds[2][1],
                    bins=', '.join([str(b) for b in self.nbins]),
                    shiftvec=self.shiftvec.tolist())
