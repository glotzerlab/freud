# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import io
import numpy as np
try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False


def ax_to_bytes(ax):
    """Helper function to convert figure to png file.

    Args:
        ax (:class:`matplotlib.axes.Axes`): axes object to plot.

    Returns:
        bytes: Byte representation of the diagram in png format.
    """
    if ax is None:
        return None
    f = io.BytesIO()
    # Sets an Agg backend so this figure can be rendered
    fig = ax.figure
    FigureCanvasAgg(fig)
    fig.savefig(f, format='png')
    fig.clf()
    return f.getvalue()


def bar_plot(x, height, title=None, xlabel=None, ylabel=None, ax=None):
    """Helper function to draw a bar graph.

    Args:
        x (list): x values of the bar graph.
        height (list): Height values corresponding to :code:`x`.
        title (str): Title of the graph. (Default value = :code:`None`).
        xlabel (str): Label of x axis. (Default value = :code:`None`).
        ylabel (str): Label of y axis. (Default value = :code:`None`).
        ax (:class:`matplotlib.axes.Axes`): axes object to plot.
            If :code:`None`, make a new axes and figure object.
            (Default value = :code:`None`).

    Returns:
        :class:`matplotlib.axes.Axes`: axes object with the diagram.
    """
    if not MATPLOTLIB:
        return None

    if ax is None:
        fig = Figure()
        ax = fig.subplots()

    ax.bar(x=x, height=height)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    return ax


def clusters_plot(keys, freqs, num_clusters_to_plot=10, ax=None):
    """Helper function to plot most frequent clusters in a bar graph.

    Args:
        keys (list): Cluster keys.
        freqs (list): Number of particles in each clusters.
        num_clusters_to_plot (unsigned int): Number of largest clusters to
            plot.
        ax (:class:`matplotlib.axes.Axes`): axes object to plot.
            If :code:`None`, make a new axes and figure object.
            (Default value = :code:`None`).

    Returns:
        :class:`matplotlib.axes.Axes`: axes object with the diagram.
    """
    count_sorted = sorted([(freq, key)
                          for key, freq in zip(keys, freqs)],
                          key=lambda x: -x[0])
    sorted_freqs = [i[0] for i in count_sorted[:num_clusters_to_plot]]
    sorted_keys = [str(i[1]) for i in count_sorted[:num_clusters_to_plot]]
    return bar_plot(sorted_keys, sorted_freqs, title="Cluster Frequency",
                    xlabel="Keys of {} largest clusters (total clusters: "
                           "{})".format(len(sorted_freqs), len(freqs)),
                    ylabel="Number of particles", ax=ax)


def line_plot(x, y, title=None, xlabel=None, ylabel=None, ax=None):
    """Helper function to draw a line graph.

    Args:
        x (list): x values of the line graph.
        y (list): y values corresponding to :code:`x`.
        title (str): Title of the graph. (Default value = :code:`None`).
        xlabel (str): Label of x axis. (Default value = :code:`None`).
        ylabel (str): Label of y axis. (Default value = :code:`None`).
        ax (:class:`matplotlib.axes.Axes`): axes object to plot.
            If :code:`None`, make a new axes and figure object.
            (Default value = :code:`None`).

    Returns:
        :class:`matplotlib.axes.Axes`: axes object with the diagram.
    """
    if not MATPLOTLIB:
        return None

    if ax is None:
        fig = Figure()
        ax = fig.subplots()

    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def histogram_plot(values, title=None, xlabel=None, ylabel=None, ax=None):
    """Helper function to draw a histogram graph.

    Args:
        values (list): values of the histogram.
        title (str): Title of the graph. (Default value = :code:`None`).
        xlabel (str): Label of x axis. (Default value = :code:`None`).
        ylabel (str): Label of y axis. (Default value = :code:`None`).
        ax (:class:`matplotlib.axes.Axes`): axes object to plot.
            If :code:`None`, make a new axes and figure object.
            (Default value = :code:`None`).

    Returns:
        :class:`matplotlib.axes.Axes`: axes object with the diagram.
    """
    if not MATPLOTLIB:
        return None

    if ax is None:
        fig = Figure()
        ax = fig.subplots()

    ax.hist(values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def pmft_plot(pmft, ax=None):
    """Helper function to draw 2D PMFT diagram.

    Args:
        pmft (:class:`freud.pmft.PMFTXY2D`):
            PMFTXY2D instance.
        ax (:class:`matplotlib.axes.Axes`): axes object to plot.
            If :code:`None`, make a new axes and figure object.
            (Default value = :code:`None`).

    Returns:
        :class:`matplotlib.axes.Axes`: axes object with the diagram.
    """
    if not MATPLOTLIB:
        return None
    try:
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
        from matplotlib.colorbar import Colorbar
    except ImportError:
        return None
    else:
        # Plot figures
        if ax is None:
            fig = Figure()
            ax = fig.subplots()

        pmft_arr = np.copy(pmft.PMFT)
        pmft_arr[np.isinf(pmft_arr)] = np.nan

        xlims = (pmft.X[0], pmft.X[-1])
        ylims = (pmft.Y[0], pmft.Y[-1])
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.xaxis.set_ticks([i for i in range(int(xlims[0]), int(xlims[1]+1))])
        ax.yaxis.set_ticks([i for i in range(int(ylims[0]), int(ylims[1]+1))])
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_title('PMFT')

        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="7%", pad="10%")

        im = ax.imshow(np.flipud(pmft_arr),
                       extent=[xlims[0], xlims[1], ylims[0], ylims[1]],
                       interpolation='nearest', cmap='viridis',
                       vmin=-2.5, vmax=3.0)

        cb = Colorbar(cax, im)
        cb.set_label(r"$k_B T$")

        return ax


def density_plot(density, box, ax=None):
    R"""Helper function to plot density diagram.

    Args:
        density (:math:`\left(N_x, N_y\right)` :class:`numpy.ndarray`):
            Array containing density.
        box (:class:`freud.box.Box`):
            Simulation box.
        ax (:class:`matplotlib.axes.Axes`): axes object to plot.
            If :code:`None`, make a new axes and figure object.
            (Default value = :code:`None`).

    Returns:
        :class:`matplotlib.axes.Axes`: axes object with the diagram.
    """
    if not MATPLOTLIB:
        return None
    try:
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
        from matplotlib.colorbar import Colorbar
    except ImportError:
        return None

    if ax is None:
        fig = Figure()
        ax = fig.subplots()

    xlims = (-box.Lx/2, box.Lx/2)
    ylims = (-box.Ly/2, box.Ly/2)

    ax.set_title('Gaussian Density')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="7%", pad="10%")

    im = ax.imshow(density.T, extent=[xlims[0], xlims[1], ylims[0], ylims[1]])

    cb = Colorbar(cax, im)
    cb.set_label("Density")

    return ax


def voronoi_plot(box, polytopes, ax=None, color_by_sides=True, cmap=None):
    """Helper function to draw 2D Voronoi diagram.

    Args:
        box (:class:`freud.box.Box`):
            Simulation box.
        polytopes (:class:`numpy.ndarray`):
            Array containing Voronoi polytope vertices.
        ax (:class:`matplotlib.axes.Axes`): axes object to plot.
            If :code:`None`, make a new axes and figure object.
            (Default value = :code:`None`).
        color_by_sides (bool):
            If :code:`True`, color cells by the number of sides.
            If :code:`False`, random colors are used for each cell.
            (Default value = :code:`True`)
        cmap (str):
            Colormap name to use (Default value = :code:`None`).

    Returns:
        :class:`matplotlib.axes.Axes`: axes object with the diagram.
    """
    if not MATPLOTLIB:
        return None
    try:
        from matplotlib import cm
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Polygon
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
        from matplotlib.colorbar import Colorbar
    except ImportError:
        return None

    if ax is None:
        fig = Figure()
        ax = fig.subplots()

    # Draw Voronoi polytopes
    patches = [Polygon(poly[:, :2]) for poly in polytopes]
    patch_collection = PatchCollection(patches, edgecolors='black', alpha=0.4)

    if color_by_sides:
        colors = np.array([len(poly) for poly in polytopes])
    else:
        colors = np.random.permutation(np.arange(len(patches)))

    cmap = cm.get_cmap('Set1' if cmap is None else cmap,
                       np.unique(colors).size)
    bounds = np.arange(np.min(colors), np.max(colors)+1)

    patch_collection.set_array(np.array(colors)-0.5)
    patch_collection.set_cmap(cmap)
    patch_collection.set_clim(bounds[0]-0.5, bounds[-1]+0.5)
    ax.add_collection(patch_collection)

    # Draw box
    corners = [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]
    corners.append(corners[0])  # Need to copy this so that the box is closed.
    corners = box.make_absolute(corners)[:, :2]
    ax.plot(corners[:, 0], corners[:, 1], color='k')

    # Set title, limits, aspect
    ax.set_title('Voronoi Diagram')
    ax.set_xlim((np.min(corners[:, 0]), np.max(corners[:, 0])))
    ax.set_ylim((np.min(corners[:, 1]), np.max(corners[:, 1])))
    ax.set_aspect('equal', 'datalim')

    # Add colorbar for number of sides
    if color_by_sides:
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="7%", pad="10%")
        cb = Colorbar(cax, patch_collection)
        cb.set_label("Number of sides")
        cb.set_ticks(bounds)
    return ax
