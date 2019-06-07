# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

# Methods used throughout freud for convenience

import logging
import numpy as np
import freud.box

logger = logging.getLogger(__name__)


def convert_array(array, dimensions=None, dtype=np.float32):
    """Function which takes a given array, checks the dimensions,
    and converts to a supplied dtype and/or makes the array
    contiguous.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    Args:
        array (:class:`numpy.ndarray`): Array to check and convert.
        dimensions (int): Expected dimensions of the array. If 'None',
            no dimensionality check will be done (Default value = 'None').
        dtype: code:`dtype` to convert the array to if :code:`array.dtype`
            is different. If `None`, :code:`dtype` will not be changed
            (Default value = `numpy.float32`).

    Returns:
        :class:`numpy.ndarray`: Array.
    """
    array = np.asarray(array)

    if dimensions is not None and array.ndim != dimensions:
        raise TypeError("array.ndim = {}; expected ndim = {}".format(
            array.ndim, dimensions))
    return np.require(array, dtype=dtype, requirements=['C'])


def convert_box(box):
    """Function which takes a box-like object and attempts to convert it to
    :class:`freud.box.Box`. Existing :class:`freud.box.Box` objects are
    used directly.

    .. moduleauthor:: Bradley Dice <bdice@bradleydice.com>

    Args:
        box (box-like object (see :meth:`freud.box.Box.from_box`)): Box to
            check and convert if needed.

    Returns:
        :class:`freud.box.Box`: freud box.
    """
    if not isinstance(box, freud.box.Box):
        try:
            box = freud.box.Box.from_box(box)
        except ValueError:
            raise
    return box


def bar_plot(x, height, title=None, xlabel=None, ylabel=None):
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import io
    except ImportError:
        return None
    else:
        fig = Figure()
        ax = fig.subplots()
        ax.bar(x=x, height=height)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(x)
        f = io.BytesIO()
        canvas = FigureCanvasAgg(fig) # noqa F841
        fig.savefig(f, format='png')
        return f.getvalue()


def histogram_plot(bins, values, title=None, xlabel=None, ylabel=None):
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import io
    except ImportError:
        return None
    else:
        fig = Figure()
        ax = fig.subplots()
        ax.hist(x=values, bins=bins)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        f = io.BytesIO()
        canvas = FigureCanvasAgg(fig) # noqa F841
        fig.savefig(f, format='png')
        return f.getvalue()


def line_plot(x, y, title=None, xlabel=None, ylabel=None):
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import io
    except ImportError:
        return None
    else:
        fig = Figure()
        ax = fig.subplots()
        ax.plot(x, y)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        f = io.BytesIO()
        canvas = FigureCanvasAgg(fig) # noqa F841
        fig.savefig(f, format='png')
        return f.getvalue()


def make_polygon(sides, radius=1):
    thetas = np.linspace(0, 2*np.pi, sides+1)[:sides]
    vertices = np.array([[radius*np.sin(theta), radius*np.cos(theta)]
                         for theta in thetas])
    return vertices


def pmft_plot(pmft):
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.colors import Normalize
        from matplotlib.cm import viridis
        import io
        from scipy.ndimage.filters import gaussian_filter
    except ImportError:
        return None
    else:
        pmft_arr = np.copy(pmft.PMFT)

        # Do some simple post-processing for plotting purposes
        pmft_arr[np.isinf(pmft_arr)] = np.nan
        dx = (2.0 * 3.0) / pmft.n_bins_X
        dy = (2.0 * 3.0) / pmft.n_bins_Y
        nan_arr = np.where(np.isnan(pmft_arr))
        for i in range(pmft.n_bins_X):
            x = -3.0 + dx * i
            for j in range(pmft.n_bins_Y):
                y = -3.0 + dy * j
                if ((x*x + y*y < 1.5) and (np.isnan(pmft_arr[j, i]))):
                    pmft_arr[j, i] = 10.0
        w = int(2.0 * pmft.n_bins_X / (2.0 * 3.0))
        center = int(pmft.n_bins_X / 2)

        # Get the center of the histogram bins
        pmft_smooth = gaussian_filter(pmft_arr, 1)
        pmft_image = np.copy(pmft_smooth)
        pmft_image[nan_arr] = np.nan
        pmft_smooth = pmft_smooth[center-w:center+w, center-w:center+w]
        pmft_image = pmft_image[center-w:center+w, center-w:center+w]
        x = pmft.X
        y = pmft.Y
        reduced_x = x[center-w:center+w]
        reduced_y = y[center-w:center+w]

        # Plot figures
        fig = Figure(figsize=(12, 5), facecolor='white')
        values = [-2, -1, 0, 2]
        norm = Normalize(vmin=-2.5, vmax=3.0)
        n_values = [norm(i) for i in values]
        colors = viridis(n_values)
        colors = colors[:, :3]
        # verts = make_polygon(sides=6, radius=0.6204)
        lims = (-2, 2)
        ax0 = fig.add_subplot(1, 2, 1)
        ax1 = fig.add_subplot(1, 2, 2)
        for ax in (ax0, ax1):
            # commented out since in general, particles are not hexagonal
            # ax.contour(reduced_x, reduced_y, pmft_smooth,
            #            [9, 10], colors='black')
            # ax.contourf(reduced_x, reduced_y, pmft_smooth,
            #             [9, 10], hatches='X', colors='none')
            # ax.plot(verts[:,0], verts[:,1], color='black', marker=',')
            # ax.fill(verts[:,0], verts[:,1], color='black')
            ax.set_aspect('equal')
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.xaxis.set_ticks([i for i in range(lims[0], lims[1]+1)])
            ax.yaxis.set_ticks([i for i in range(lims[0], lims[1]+1)])
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$y$')

        ax0.set_title('PMFT Heat Map')
        im = ax0.imshow(np.flipud(pmft_image),
                        extent=[lims[0], lims[1], lims[0], lims[1]],
                        interpolation='nearest', cmap='viridis',
                        vmin=-2.5, vmax=3.0)
        ax1.set_title('PMFT Contour Plot')
        ax1.contour(reduced_x, reduced_y, pmft_smooth,
                    [-2, -1, 0, 2], colors=colors)

        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.1, 0.02, 0.8])
        fig.colorbar(im, cax=cbar_ax)
        f = io.BytesIO()
        canvas = FigureCanvasAgg(fig) # noqa F841
        fig.savefig(f, format='png')
        return f.getvalue()


def draw_voronoi(box, cells, color_by_sides=False):
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib import cm
        import io
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Polygon, Rectangle
    except ImportError:
        return None
    fig = Figure()
    ax = fig.subplots()

    # Draw Voronoi cells
    patches = [Polygon(cell[:, :2]) for cell in cells]
    patch_collection = PatchCollection(patches, edgecolors='black', alpha=0.4)
    cmap = cm.Set1

    if color_by_sides:
        colors = [len(cell) for cell in cells]
    else:
        colors = np.random.permutation(np.arange(len(patches)))

    cmap = cm.get_cmap('Set1', np.unique(colors).size)
    bounds = np.array(range(min(colors), max(colors)+2))

    patch_collection.set_array(np.array(colors))
    patch_collection.set_cmap(cmap)
    patch_collection.set_clim(bounds[0], bounds[-1])
    ax.add_collection(patch_collection)

    ax.set_title('Voronoi Diagram')
    ax.set_xlim((-box.Lx/2, box.Lx/2))
    ax.set_ylim((-box.Ly/2, box.Ly/2))

    # Set equal aspect and draw box
    ax.set_aspect('equal', 'datalim')
    box_patch = Rectangle([-box.Lx/2, -box.Ly/2], box.Lx,
                          box.Ly, alpha=1, fill=None)
    ax.add_patch(box_patch)

    # Show colorbar for number of sides
    if color_by_sides:
        cb = fig.colorbar(patch_collection, ax=ax,
                          ticks=bounds, boundaries=bounds)
        cb.set_ticks(cb.formatter.locs + 0.5)
        cb.set_ticklabels((cb.formatter.locs - 0.5).astype('int'))
        cb.set_label("Number of sides", fontsize=12)
    f = io.BytesIO()
    canvas = FigureCanvasAgg(fig) # noqa F841
    fig.savefig(f, format='png')
    return f.getvalue()
