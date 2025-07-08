# Copyright (c) 2010-2025 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

r"""
The :class:`freud.density` module contains various classes relating to the
density of the system. These functions allow evaluation of particle
distributions with respect to other particles.
"""

from collections.abc import Sequence

import numpy as np

import freud
import freud._density
import freud.plot
import freud.util
from freud.locality import _PairCompute, _SpatialHistogram1D
from freud.util import _Compute


class CorrelationFunction(_SpatialHistogram1D):
    r"""Computes the complex pairwise correlation function.

    The correlation function is given by
    :math:`C(r) = \left\langle s^*_1(0) \cdot s_2(r) \right\rangle` between
    two sets of points :math:`p_1` (:code:`points`) and :math:`p_2`
    (:code:`query_points`) with associated values :math:`s_1` (:code:`values`)
    and :math:`s_2` (:code:`query_values`). Computing the correlation function
    results in an array of the expected (average) product of all values at a
    given radial distance :math:`r`.
    The values of :math:`r` where the correlation function is computed are
    controlled by the :code:`bins` and :code:`r_max` parameters to the
    constructor, and the spacing between the bins is given by
    :code:`dr = r_max / bins`.

    .. note::
        **Self-correlation:** It is often the case that we wish to compute the
        correlation function of a set of points with itself. If
        :code:`query_points` is the same as :code:`points`, not provided, or
        :code:`None`, we omit accumulating the self-correlation value in the
        first bin.

    Args:
        bins (unsigned int):
            The number of bins in the correlation function.
        r_max (float):
            Maximum pointwise distance to include in the calculation.
    """

    def __init__(self, bins, r_max):
        self._bins = int(bins)
        self._cpp_obj = freud._density.CorrelationFunction(self._bins, r_max)
        self.r_max = r_max
        self.is_complex = False

    def compute(
        self,
        system,
        values,
        query_points=None,
        query_values=None,
        neighbors=None,
        reset=True,
    ):
        r"""Calculates the correlation function and adds to the current
        histogram.

        Args:
            system:
                Any object that is a valid argument to
                :class:`freud.locality.NeighborQuery.from_system`.
            values ((:math:`N_{points}`) :class:`numpy.ndarray`):
                Values associated with the system points used to calculate the
                correlation function.
            query_points ((:math:`N_{query\_points}`, 3) :class:`numpy.ndarray`, optional):
                Query points used to calculate the correlation function.  Uses
                the system's points if :code:`None` (Default value =
                :code:`None`).
            query_values ((:math:`N_{query\_points}`) :class:`numpy.ndarray`, optional):
                Query values used to calculate the correlation function.  Uses
                :code:`values` if :code:`None`.  (Default value
                = :code:`None`).
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
        """  # noqa E501
        if reset:
            self.is_complex = False
            self._reset()

        nq, nlist, qargs, l_query_points, num_query_points = self._preprocess_arguments(
            system, query_points, neighbors
        )

        self.is_complex = (
            self.is_complex
            or np.any(np.iscomplex(values))
            or np.any(np.iscomplex(query_values))
        )

        values = freud.util._convert_array(
            values,
            shape=(nq.points.shape[0],),
            dtype=np.complex128,
        )
        if query_values is None:
            query_values = values
        else:
            query_values = freud.util._convert_array(
                query_values,
                shape=(l_query_points.shape[0],),
                dtype=np.complex128,
            )
        self._cpp_obj.accumulate(
            nq._cpp_obj,
            values,
            l_query_points,
            query_values,
            nlist._cpp_obj,
            qargs._cpp_obj,
        )
        return self

    @_Compute._computed_property
    def correlation(self):
        """(:math:`N_{bins}`) :class:`numpy.ndarray`: Expected (average)
        product of all values at a given radial distance."""
        output = self._cpp_obj.getCorrelation().toNumpyArray()
        return output if self.is_complex else np.real(output)

    @property
    def bins(self):
        """int: Number of bins used in the calculation."""
        return self._cpp_obj.getNBins()

    def __repr__(self):
        return (
            f"freud.density.{type(self).__name__}(bins={self.nbins}, "
            f"r_max={self.r_max})"
        )

    def plot(self, ax=None):
        """Plot complex correlation function.

        Args:
            ax (:class:`matplotlib.axes.Axes`, optional): Axis to plot on. If
                :code:`None`, make a new figure and axis.
                (Default value = :code:`None`)

        Returns:
            (:class:`matplotlib.axes.Axes`): Axis with the plot.
        """

        return freud.plot.line_plot(
            self.bin_centers,
            np.real(self.correlation),
            title="Correlation Function",
            xlabel=r"$r$",
            ylabel=r"$\operatorname{Re}(C(r))$",
            ax=ax,
        )

    def _repr_png_(self):
        try:
            return freud.plot._ax_to_bytes(self.plot())
        except (AttributeError, ImportError):
            return None


class GaussianDensity(_Compute):
    r"""Computes the density of a system on a grid.

    Replaces particle positions with a Gaussian blur and calculates the
    contribution from each to the proscribed grid based upon the distance of
    the grid cell from the center of the Gaussian. The weights for the
    Gaussians could be additionally specified in the compute method. The
    convolution of the weights with the Gaussians is calculated in this case:

    .. math::

        p(\vec{r}) = \sum_i \frac{1}{2\pi \sigma^2}
        \exp \left(-\frac{(\vec{r}-\vec{r}_i)^2}{2\sigma^2}\right) p_i

    The resulting data is a regular grid of particle densities or
    convolved parameter that can be used in standard algorithms
    requiring evenly spaced point, such as Fast Fourier Transforms. The
    dimensions of the grid are set in the constructor, and can either be set
    equally for all dimensions or for each dimension independently.

    Args:
        width (int or Sequence[int]):
            The number of bins to make the grid in each dimension (identical
            in all dimensions if a single integer value is provided).
        r_max (float):
            Distance over which to blur.
        sigma (float):
            Sigma parameter for Gaussian.
    """

    def __init__(self, width, r_max, sigma):
        if isinstance(width, int):
            width_vector = np.array([width, width, width])
        elif isinstance(width, Sequence) and len(width) == 2:
            width_vector = np.array([width[0], width[1], 1])
        elif isinstance(width, Sequence) and len(width) == 3:
            width_vector = np.array([width[0], width[1], width[2]])
        else:
            error_message = (
                "The width must be either a number of bins or a "
                "sequence indicating the widths in each spatial "
                "dimension (length 2 in 2D, length 3 in 3D)."
            )
            raise ValueError(error_message)

        self._cpp_obj = freud._density.make_gaussian_density(
            width_vector[0], width_vector[1], width_vector[2], r_max, sigma
        )

    @_Compute._computed_property
    def box(self):
        """:class:`freud.box.Box`: Box used in the calculation."""
        return freud.box.BoxFromCPP(self._cpp_obj.getBox())

    def compute(self, system, values=None):
        r"""Calculates the Gaussian blur for the specified points.

        Args:
            system:
                Any object that is a valid argument to
                :class:`freud.locality.NeighborQuery.from_system`.
            values ((:math:`N_{points}`) :class:`numpy.ndarray`):
                Values associated with the system points used to calculate the
                convolution. Calculates Gaussian blur (equivalent to providing
                a value of 1 for every point) if :code:`None`. (Default value
                = :code:`None`).
        """
        nq = freud.locality.NeighborQuery.from_system(system)

        if values is not None:
            l_values = freud.util._convert_array(values, shape=(nq.points.shape[0],))
        else:
            l_values = None

        self._cpp_obj.compute(nq._cpp_obj, l_values)
        return self

    @_Compute._computed_property
    def density(self):
        """(:math:`w_x`, :math:`w_y`, :math:`w_z`) :class:`numpy.ndarray`: The
        grid with the Gaussian density contributions from each point."""
        if self.box.is2D:
            return np.squeeze(self._cpp_obj.density.toNumpyArray())
        return self._cpp_obj.density.toNumpyArray()

    @property
    def r_max(self):
        """float: Distance over which to blur."""
        return self._cpp_obj.getRMax()

    @property
    def sigma(self):
        """float: Sigma parameter for Gaussian."""
        return self._cpp_obj.getSigma()

    @property
    def width(self):
        """tuple[int]: The number of bins in the grid in each dimension
        (identical in all dimensions if a single integer value is provided)."""
        return self._cpp_obj.getWidth()

    def __repr__(self):
        return (
            f"freud.density.{type(self).__name__}({self.width}, "
            f"{self.r_max}, {self.sigma})"
        )

    def plot(self, ax=None):
        """Plot Gaussian Density.

        Args:
            ax (:class:`matplotlib.axes.Axes`, optional): Axis to plot on. If
                :code:`None`, make a new figure and axis.
                (Default value = :code:`None`)

        Returns:
            (:class:`matplotlib.axes.Axes`): Axis with the plot.
        """

        if not self.box.is2D:
            return None
        return freud.plot.density_plot(self.density, self.box, ax=ax)

    def _repr_png_(self):
        try:
            return freud.plot._ax_to_bytes(self.plot())
        except (AttributeError, ImportError):
            return None


class SphereVoxelization(_Compute):
    r"""Computes a grid of voxels occupied by spheres.

    This class constructs a grid of voxels. From a given set of points and a
    desired radius, a set of spheres are created. The voxels are assigned a
    value of 1 if their center is contained in one or more spheres and 0
    otherwise. The dimensions of the grid are set in the constructor, and can
    either be set equally for all dimensions or for each dimension
    independently.

    Args:
        width (int or Sequence[int]):
            The number of bins to make the grid in each dimension (identical
            in all dimensions if a single integer value is provided).
        r_max (float):
            Sphere radius.
    """

    def __init__(self, width, r_max):
        if isinstance(width, int):
            width_vector = np.array([width, width, width])
        elif isinstance(width, Sequence) and len(width) == 2:
            width_vector = np.array([width[0], width[1], 1])
        elif isinstance(width, Sequence) and len(width) == 3:
            width_vector = np.array([width[0], width[1], width[2]])
        else:
            error_message = (
                "The width must be either a number of bins or a "
                "sequence indicating the widths in each spatial "
                "dimension (length 2 in 2D, length 3 in 3D)."
            )
            raise ValueError(error_message)

        self._cpp_obj = freud._density.make_sphere_voxelization(
            width_vector[0], width_vector[1], width_vector[2], r_max
        )

    @_Compute._computed_property
    def box(self):
        """:class:`freud.box.Box`: Box used in the calculation."""
        return freud.box.BoxFromCPP(self._cpp_obj.box)

    def compute(self, system):
        r"""Calculates the voxelization of spheres about the specified points.

        Args:
            system:
                Any object that is a valid argument to
                :class:`freud.locality.NeighborQuery.from_system`.
        """
        nq = freud.locality.NeighborQuery.from_system(system)
        self._cpp_obj.compute(nq._cpp_obj)
        return self

    @_Compute._computed_property
    def voxels(self):
        """(:math:`w_x`, :math:`w_y`, :math:`w_z`) :class:`numpy.ndarray`: The
        voxel grid indicating overlap with the computed spheres."""
        data = self._cpp_obj.voxels.toNumpyArray()
        if self.box.is2D:
            return np.squeeze(data)
        return data

    @property
    def r_max(self):
        """float: Sphere radius used for voxelization."""
        return self._cpp_obj.getRMax()

    @property
    def width(self):
        """tuple[int]: The number of bins in the grid in each dimension
        (identical in all dimensions if a single integer value is provided)."""
        return self._cpp_obj.getWidth()

    def __repr__(self):
        return f"freud.density.{type(self).__name__}({self.width}, {self.r_max})"

    def plot(self, ax=None):
        """Plot voxelization.

        Args:
            ax (:class:`matplotlib.axes.Axes`, optional): Axis to plot on. If
                :code:`None`, make a new figure and axis.
                (Default value = :code:`None`)

        Returns:
            (:class:`matplotlib.axes.Axes`): Axis with the plot.
        """

        if not self.box.is2D:
            return None
        return freud.plot.density_plot(self.voxels, self.box, ax=ax)

    def _repr_png_(self):
        try:
            return freud.plot._ax_to_bytes(self.plot())
        except (AttributeError, ImportError):
            return None


class LocalDensity(_PairCompute):
    r"""Computes the local density around a particle.

    The density of the local environment is computed and averaged for a given
    set of query points in a sea of data points. Providing the same points
    calculates them against themselves. Computing the local density results in
    an array listing the value of the local density around each query
    point. Also available is the number of neighbors for each query point,
    giving the user the ability to count the number of particles in that
    region. Note that the computed density is essentially a number density
    (allowing for fractional values as described below). If particles
    have a specific volume, a volume density can be computed by simply
    multiplying the number density by the volume of the particles.

    In order to provide sufficiently smooth data, data points can be
    fractionally counted towards the density. Rather than perform
    compute-intensive area (volume) overlap calculations to
    determine the exact amount of overlap area (volume), the LocalDensity class
    performs a simple linear interpolation relative to the centers of the data
    points.  Specifically, a point is counted as one neighbor of a given
    query point if it is entirely contained within the :code:`r_max`, half
    of a neighbor if the distance to its center is exactly :code:`r_max`, and
    zero if its center is a distance greater than or equal to :code:`r_max +
    diameter` from the query point's center. Graphically, this looks like:

    .. image:: images/density.png

    Args:
        r_max (float):
            Maximum distance over which to calculate the density.
        diameter (float):
            Diameter of particle circumsphere.
    """

    def __init__(self, r_max, diameter):
        self._cpp_obj = freud._density.LocalDensity(r_max, diameter)

    @property
    def r_max(self):
        """float: Maximum distance over which to calculate the density."""
        return self._cpp_obj.getRMax()

    @property
    def diameter(self):
        """float: Diameter of particle circumsphere."""
        return self._cpp_obj.getDiameter()

    @_Compute._computed_property
    def box(self):
        """:class:`freud.box.Box`: Box used in the calculation."""
        return freud.box.BoxFromCPP(self._cpp_obj.box)

    def compute(self, system, query_points=None, neighbors=None):
        r"""Calculates the local density for the specified points.

        Example::

            >>> import freud
            >>> box, points = freud.data.make_random_system(10, 100, seed=0)
            >>> # Compute Local Density
            >>> ld = freud.density.LocalDensity(r_max=3, diameter=0.05)
            >>> ld.compute(system=(box, points))
            freud.density.LocalDensity(...)

        Args:
            system:
                Any object that is a valid argument to
                :class:`freud.locality.NeighborQuery.from_system`.
            query_points ((:math:`N_{query\_points}`, 3) :class:`numpy.ndarray`, optional):
                Query points used to calculate the correlation function. Uses
                the system's points if :code:`None` (Default
                value = :code:`None`).
            neighbors (:class:`freud.locality.NeighborList` or dict, optional):
                Either a :class:`NeighborList <freud.locality.NeighborList>` of
                neighbor pairs to use in the calculation, or a dictionary of
                `query arguments
                <https://freud.readthedocs.io/en/stable/topics/querying.html>`_
                (Default value: None).
        """  # noqa E501

        nq, nlist, qargs, l_query_points, num_query_points = self._preprocess_arguments(
            system, query_points, neighbors
        )
        self._cpp_obj.compute(
            nq._cpp_obj,
            l_query_points,
            num_query_points,
            nlist._cpp_obj,
            qargs._cpp_obj,
        )
        return self

    @property
    def default_query_args(self):
        """The default query arguments are
        :code:`{'mode': 'ball', 'r_max': self.r_max + 0.5*self.diameter}`."""
        return dict(mode="ball", r_max=self.r_max + 0.5 * self.diameter)

    @_Compute._computed_property
    def density(self):
        """(:math:`N_{points}`) :class:`numpy.ndarray`: Density of points per
        query point."""
        return self._cpp_obj.density.toNumpyArray()

    @_Compute._computed_property
    def num_neighbors(self):
        """(:math:`N_{points}`) :class:`numpy.ndarray`: Number of neighbor
        points for each query point."""
        return self._cpp_obj.num_neighbors.toNumpyArray()

    def __repr__(self):
        return (
            f"freud.density.{type(self).__name__}(r_max={self.r_max}, "
            f"diameter={self.diameter})"
        )


class RDF(_SpatialHistogram1D):
    r"""Computes the RDF :math:`g \left( r \right)` for supplied data.

    Note that the RDF is defined strictly according to the pair correlation
    function, i.e.

    .. math::

        g(r) = V\frac{N-1}{N} \langle \delta(r) \rangle

    In the thermodynamic limit, the fraction tends to unity and the limiting
    behavior of :math:`\lim_{r \to \infty} g(r)=1` is recovered. However, for
    very small systems the long range behavior of the radial distribution will
    instead tend to :math:`\frac{N-1}{N}`. In small systems, where this
    deviation is noticeable, the ``normalization_mode`` argument may be used to
    rescale the results and force the long range behavior to 1. Note that this
    option will have little to no effect on larger systems (for example, for
    systems of 100 particles the RDF will differ by 1%).

    .. note::
        For correct normalization behavior, let the set of points be either: 1)
        the same as the set of query points or 2) completely disjoint from the
        set of query points (points shouldn't contain any particles in query
        points).

    .. note::
        For correct normalization behavior when using
        ``normalization_mode='finite_size'``, the ``points`` _must_ be the same
        as the ``query_points`` and ``exclude_ii`` must be set to ``False``.

    .. note::
        **2D:** :class:`freud.density.RDF` properly handles 2D boxes.
        The points must be passed in as :code:`[x, y, 0]`.

    Args:
        bins (unsigned int):
            The number of bins in the RDF.
        r_max (float):
            Maximum interparticle distance to include in the calculation.
        r_min (float, optional):
            Minimum interparticle distance to include in the calculation
            (Default value = :code:`0`).
        normalization_mode (str, optional):
            There are two valid string inputs for this argument. The first
            option, ``exact``, handles the normalization as shown mathematically
            at the beginning of this class's docstring. The other option,
            ``finite_size``, adds an extra rescaling factor of
            :math:`\frac{N_{query\_points}}{N_{query\_ponts} - 1}` so the RDF
            values will tend to 1 at large :math:`r` for small systems (Default
            value = :code:`'exact'`).
    """

    def __init__(self, bins, r_max, r_min=0, normalization_mode="exact"):
        self._cpp_obj = freud._density.RDF(bins, r_max, r_min)
        self.mode = normalization_mode

        # r_max is left as an attribute rather than a property for now
        # since that change needs to happen at the _SpatialHistogram level
        # for multiple classes.
        self.r_max = r_max

    @property
    def mode(self):
        if self._cpp_obj.mode == freud._density.NormalizationMode.exact:
            return "exact"
        if self._cpp_obj.mode == freud._density.NormalizationMode.finite_size:
            return "finite_size"
        msg = f"Unknown normalization mode {self._cpp_obj.mode} set!"
        raise ValueError(msg)

    @mode.setter
    def mode(self, value):
        if value == "exact":
            self._cpp_obj.mode = freud._density.NormalizationMode.exact
        elif value == "finite_size":
            self._cpp_obj.mode = freud._density.NormalizationMode.finite_size
        else:
            msg = "An invalid normalization mode was provided."
            raise ValueError(msg)

    def compute(self, system, query_points=None, neighbors=None, reset=True):
        r"""Calculates the RDF and adds to the current RDF histogram.

        Args:
            system:
                Any object that is a valid argument to
                :class:`freud.locality.NeighborQuery.from_system`.
            query_points ((:math:`N_{query\_points}`, 3) :class:`numpy.ndarray`, optional):
                Query points used to calculate the RDF. Uses the system's
                points if :code:`None` (Default value =
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
        """  # noqa E501
        if reset:
            self._reset()

        nq, nlist, qargs, l_query_points, num_query_points = self._preprocess_arguments(
            system, query_points, neighbors
        )

        self._cpp_obj.accumulateRDF(
            nq._cpp_obj, l_query_points, nlist._cpp_obj, qargs._cpp_obj
        )
        return self

    @_Compute._computed_property
    def rdf(self):
        """(:math:`N_{bins}`,) :class:`numpy.ndarray`: Histogram of RDF
        values."""
        return self._cpp_obj.getRDF().toNumpyArray()

    @_Compute._computed_property
    def n_r(self):
        """(:math:`N_{bins}`,) :class:`numpy.ndarray`: Histogram of cumulative
        bin_counts values. More precisely, :code:`n_r[i]` is the average number
        of points contained within a ball of radius :code:`bin_edges[i+1]`
        centered at a given :code:`query_point` averaged over all
        :code:`query_points` in the last call to :meth:`~.compute`."""
        return self._cpp_obj.getNr().toNumpyArray()

    def __repr__(self):
        return (
            f"freud.density.{type(self).__name__}(bins={len(self.bin_centers)}, "
            f"r_max={self.bounds[1]}, r_min={self.bounds[0]})"
        )

    def plot(self, ax=None):
        """Plot radial distribution function.

        Args:
            ax (:class:`matplotlib.axes.Axes`, optional): Axis to plot on. If
                :code:`None`, make a new figure and axis.
                (Default value = :code:`None`)

        Returns:
            (:class:`matplotlib.axes.Axes`): Axis with the plot.
        """

        return freud.plot.line_plot(
            self.bin_centers,
            self.rdf,
            title="RDF",
            xlabel=r"$r$",
            ylabel=r"$g(r)$",
            ax=ax,
        )

    def _repr_png_(self):
        try:
            return freud.plot._ax_to_bytes(self.plot())
        except (AttributeError, ImportError):
            return None
