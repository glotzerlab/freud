import numpy as np
import freud


def make_box_and_random_points(box_size, num_points, is2D=False, seed=0):
    R"""Helper function to make random points with a box.

    This function has a side effect that it will set the random seed of numpy.

    .. moduleauthor:: Jin Soo Ihm <jinihm@umich.edu>

    Args:
        box_size (float): Size of box.
        num_points (int): Number of points.
        is2D (bool): If true, points and box are in a 2D system.
            (Default value = False).
        seed (int): Random seed to use. (Default value = 0).

    Returns:
        tuple (:class:`freud.box.Box`, (:math:`\left(num_points`, 3\right)` :class:`numpy.ndarray`):
            Generated box and points.
    """  # noqa: E501
    np.random.seed(seed)
    points = np.random.random_sample((num_points, 3)).astype(np.float32) \
        * box_size - box_size/2

    if is2D is True:
        box = freud.box.Box.square(box_size)
        points[:, 2] = 0
    else:
        box = freud.box.Box.cube(box_size)

    return box, points


def make_alternating_lattice(lattice_size, angle=0, extra_shell=2):
    R"""Make 2D integer lattice of alternating set of points.

    Setting extra_shell to 1 will give 4 neighboring points in points_2 at
    distance 1 for each point in points_1. Setting extra_shell to 2 will give
    8 more neighboring points in points_2 at distance :math:`\sqrt{5}` for each
    point in points_1 and so on.

    .. moduleauthor:: Jin Soo Ihm <jinihm@umich.edu>

    Args:
        lattice_size (int): Size of lattice for points_1.
        angle (float): Angle to rotate the lattice.
            (Default value = 0)
        extra_shell (int): Extra shell of points_2 to wrap around points_1.
            (Default value = 2)

    Returns:
        tuple ((:math:`\left(N_1`, 3\right)` :class:`numpy.ndarray`),
            (:math:`\left(N_2`, 3\right)` :class:`numpy.ndarray`):
            Generated sets of points.
    """  # noqa: E501
    points_1 = []
    points_2 = []

    # Sometimes, we need to rotate the points to avoid the
    # boundary of bins. Due to numeric precision, boundaries are not
    # handled well in a convoluted input like this.
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0],
                                [0, 0, 1]])

    # make alternating lattice of points_2 and points_1
    for i in range(-extra_shell, lattice_size + extra_shell):
        for j in range(-extra_shell, lattice_size + extra_shell):
            p = np.array([i, j, 0])
            p = rotation_matrix.dot(p)
            if (i + j) % 2 == 0:
                points_2.append(p)
            else:
                if 0 <= i < lattice_size and 0 <= j < lattice_size:
                    points_1.append(p)

    points_1 = np.array(points_1)
    points_2 = np.array(points_2)
    return points_1, points_2


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
