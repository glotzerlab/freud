import io
import numpy as np
import freud
try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False


def ax_to_bytes(ax):
    f = io.BytesIO()
    # Sets an Agg backend so this figure can be rendered
    fig = ax.fig
    FigureCanvasAgg(fig)
    fig.savefig(f, format='png')
    return f.getvalue()


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
        return ax


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
        return ax


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
        from matplotlib.cm import viridis
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
        from mpl_toolkits.axes_grid1.colorbar import colorbar
    except ImportError:
        return None
    else:
        pmft_arr = np.copy(pmft.PMFT)
        pmft_arr[np.isinf(pmft_arr)] = np.nan

        # Plot figures
        fig = Figure()
        ax = fig.subplots()

        xlims = (pmft.X[0], pmft.X[-1])
        ylims = (pmft.Y[0], pmft.Y[-1])
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.xaxis.set_ticks([i for i in range(int(xlims[0]), int(xlims[1]+1))])
        ax.yaxis.set_ticks([i for i in range(int(ylims[0]), int(ylims[1]+1))])
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')

        ax.set_title('PMFT')
        im = ax.imshow(np.flipud(pmft_arr),
                       extent=[xlims[0], xlims[1], ylims[0], ylims[1]],
                       interpolation='nearest', cmap='viridis',
                       vmin=-2.5, vmax=3.0)

        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="7%", pad="10%")
        cb = colorbar(im, cax=cax)
        cb.set_label_text(r"$k_B T$")

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
