import numpy as np
import freud


def quatRandom():
    """Returns a random quaternion culled from a uniform distribution on the
    surface of a 3-sphere. Uses the Marsaglia (1972) method (a la HOOMD).
    Note that generating a random rotation via a random angle about a random
    axis of rotation is INCORRECT. See K. Shoemake, "Uniform Random Rotations,"
    1992, for a nice explanation for this.

    The output quaternion is an array of four numbers: [q0, q1, q2, q3]"""

    # np.random.uniform(low, high) gives a number from the interval [low, high)
    v1 = np.random.uniform(-1, 1)
    v2 = np.random.uniform(-1, 1)
    v3 = np.random.uniform(-1, 1)
    v4 = np.random.uniform(-1, 1)

    s1 = v1*v1 + v2*v2
    s2 = v3*v3 + v4*v4

    while (s1 >= 1.):
        v1 = np.random.uniform(-1, 1)
        v2 = np.random.uniform(-1, 1)
        s1 = v1*v1 + v2*v2

    while (s2 >= 1. or s2 == 0.):
        v3 = np.random.uniform(-1, 1)
        v4 = np.random.uniform(-1, 1)
        s2 = v3*v3 + v4*v4

    s3 = np.sqrt((1.-s1)/s2)

    return np.array([v1, v2, v3*s3, v4*s3])


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
