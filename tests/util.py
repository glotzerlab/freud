import numpy as np
import freud


def make_alternating_lattice(lattice_size, angle=0, extra_shell=2):
    """Make 2D integer lattice of alternating points and ref_points.

    .. moduleauthor:: Jin Soo Ihm <jinihm@umich.edu>

    Args:
        lattice_size (int): Size of lattice for ref_points.
        angle (float): Angle to rotate the lattice.
            (Default value = 0)
        extra_shell (int): Extra shell of points to wrap around ref_points.
            (Default value = 2)

    Returns:
        ref_points, points ((:math:`N`, 3) :class:`numpy.ndarray`):
            Generated ref_points and points
    """
    points = []
    ref_points = []

    # Sometimes, we need to rotate the points to avoid the
    # boundary of bins. Due to numeric precision, boundaries are not
    # handled well in a convoluted input like this.
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0],
                                [0, 0, 1]])

    # make alternating lattice of points and ref_points
    for i in range(-extra_shell, lattice_size + extra_shell):
        for j in range(-extra_shell, lattice_size + extra_shell):
            p = np.array([i, j, 0])
            p = rotation_matrix.dot(p)
            if (i + j) % 2 == 0:
                points.append(p)
            else:
                if 0 <= i < lattice_size and 0 <= j < lattice_size:
                    ref_points.append(p)

    ref_points = np.array(ref_points)
    points = np.array(points)
    return ref_points, points


def make_fcc(nx=1, ny=1, nz=1, scale=1.0, noise=0.0):
    """Make an FCC crystal for testing

    :param nx: Number of repeats in the x direction, default is 1
    :param ny: Number of repeats in the y direction, default is 1
    :param nz: Number of repeats in the z direction, default is 1
    :param scale: Amount to scale the unit cell by (in distance units),
        default is 1.0
    :param noise: Apply Gaussian noise with this width to particle positions
    :type nx: int
    :type ny: int
    :type nz: int
    :type scale: float
    :type noise: float
    :return: freud Box, particle positions, shape=(nx*ny*nz, 3)
    :rtype: (:class:`freud.box.Box`, :class:`np.ndarray`)
    """
    fractions = np.array([[.5, .5, 0],
                          [.5, 0, .5],
                          [0, .5, .5],
                          [0, 0, 0]], dtype=np.float32)

    fractions = np.tile(fractions[np.newaxis, np.newaxis, np.newaxis],
                        (nx, ny, nz, 1, 1))
    fractions[..., 0] += np.arange(nx)[:, np.newaxis, np.newaxis, np.newaxis]
    fractions[..., 1] += np.arange(ny)[np.newaxis, :, np.newaxis, np.newaxis]
    fractions[..., 2] += np.arange(nz)[np.newaxis, np.newaxis, :, np.newaxis]
    fractions /= [nx, ny, nz]

    box = 2*scale*np.array([nx, ny, nz], dtype=np.float32)
    positions = ((fractions - .5)*box).reshape((-1, 3))

    if noise != 0:
        positions += np.random.normal(scale=noise, size=positions.shape)

    return freud.box.Box(*box), positions


def skipIfMissing(library):
    try:
        import importlib
        import unittest
        importlib.import_module(library)
        return lambda func: func
    except ImportError:
        return unittest.skip(
            "You must have {library} installed to run this test".format(
                library=library))
