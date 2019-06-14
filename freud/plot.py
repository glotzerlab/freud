import io
import numpy as np
import freud
try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False


def bar_plot(x, height, title=None, xlabel=None, ylabel=None):
    """Helper function to draw a bar graph.

    .. moduleauthor:: Jin Soo Ihm <jinihm@umich.edu>

    Args:
        x (list): x values of the bar graph.
        height (list): Height values corresponding to :code:`x`.
        title (str): Title of the graph. (Default value = :code:`None`).
        xlabel (str): Label of x axis. (Default value = :code:`None`).
        ylabel (str): Label of y axis. (Default value = :code:`None`).

    Returns:
        bytes: Byte representation of the graph in png format if matplotlib is
        available. Otherwise :code:`None`.
    """
    if not MATPLOTLIB:
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
        # Sets an Agg backend so this figure can be rendered
        FigureCanvasAgg(fig)
        fig.savefig(f, format='png')
        return f.getvalue()


def plot_clusters(keys, freqs, num_cluster_to_plot=10):
    """Helper function to plot most frequent clusters in a bar graph.

    .. moduleauthor:: Jin Soo Ihm <jinihm@umich.edu>

    Args:
        keys (list): Cluster keys.
        freqs (list): Number of particles in each clusters.
        num_cluster_to_plot (unsigned int): Number of the most frequent
            clusters to plot.

    Returns:
        bytes: Byte representation of the graph in png format if matplotlib is
        available. Otherwise :code:`None`.
    """
    count_sorted = sorted([(freq, key)
                          for key, freq in zip(keys, freqs)],
                          key=lambda x: -x[0])
    sorted_freqs = [i[0] for i in count_sorted[:num_cluster_to_plot]]
    sorted_keys = [str(i[1]) for i in count_sorted[:num_cluster_to_plot]]
    return bar_plot(sorted_keys, sorted_freqs, title="Cluster Frequency",
                    xlabel="Keys of {} largest clusters (total clusters: "
                           "{})".format(len(sorted_freqs), len(freqs)),
                    ylabel="Number of particles")


def line_plot(x, y, title=None, xlabel=None, ylabel=None):
    """Helper function to draw a line graph.

    .. moduleauthor:: Jin Soo Ihm <jinihm@umich.edu>

    Args:
        x (list): x values of the line graph.
        y (list): y values corresponding to :code:`x`.
        title (str): Title of the graph. (Default value = :code:`None`).
        xlabel (str): Label of x axis. (Default value = :code:`None`).
        ylabel (str): Label of y axis. (Default value = :code:`None`).

    Returns:
        bytes: Byte representation of the graph in png format if matplotlib is
        available. Otherwise :code:`None`.
    """
    if not MATPLOTLIB:
        return None
    else:
        fig = Figure()
        ax = fig.subplots()
        ax.plot(x, y)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        f = io.BytesIO()
        # Sets an Agg backend so this figure can be rendered
        FigureCanvasAgg(fig)
        fig.savefig(f, format='png')
        return f.getvalue()


def pmft_plot(pmft):
    """Helper function to draw 2D PMFT diagram.

    Args:
        pmft (:class:`freud.pmft.PMFTXY2D`):
            PMFTXY2D instance.

    Returns:
        bytes: Byte representation of the diagram in png format if matplotlib
        is available. Otherwise :code:`None`.
    """
    if not MATPLOTLIB:
        return None
    try:
        from matplotlib.colors import Normalize
        from matplotlib.cm import viridis
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
        lims = (-2, 2)
        ax0 = fig.add_subplot(1, 2, 1)
        ax1 = fig.add_subplot(1, 2, 2)
        for ax in (ax0, ax1):
            # commented out since in general, particles are not hexagonal
            # ax.contour(reduced_x, reduced_y, pmft_smooth,
            #            [9, 10], colors='black')
            # ax.contourf(reduced_x, reduced_y, pmft_smooth,
            #             [9, 10], hatches='X', colors='none')
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
        # Sets an Agg backend so this figure can be rendered
        FigureCanvasAgg(fig)
        fig.savefig(f, format='png')
        return f.getvalue()


def draw_voronoi(box, cells, color_by_sides=False):
    """Helper function to draw 2D Voronoi diagram.

    Args:
        box (:class:`freud.box.Box`):
            Simulation box.
        cells (:class:`numpy.ndarray`):
            Array containing Voronoi polytope vertices.
        color_by_sides (bool):
            If :code:`True`, color cells by the number of sides.
            (Default value = :code:`False`)

    Returns:
        bytes: Byte representation of the diagram in png format if matplotlib
        is available. Otherwise :code:`None`.
    """
    if not MATPLOTLIB:
        return None
    try:
        from matplotlib import cm
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
    # Sets an Agg backend so this figure can be rendered
    FigureCanvasAgg(fig)
    fig.savefig(f, format='png')
    return f.getvalue()
