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
        ax.plot(y, x)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        f = io.BytesIO()
        canvas = FigureCanvasAgg(fig) # noqa F841
        fig.savefig(f, format='png')
        return f.getvalue()
