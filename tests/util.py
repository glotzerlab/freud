import numpy as np
import freud

def make_fcc(nx=1, ny=1, nz=1, scale=1.0, noise=0.0):
    """Make an FCC crystal for testing

    Args
    ----
    nx : int, optional
        Repeat the unit cell nx times in x direction, default=1
    ny : int, optional
        Repeat the unit cell ny times in y direction, default=1
    nz : int, optional
        Repeat the unit cell nz times in z direction, default=1
    scale : float
        Scale the unit cell by scale, default=1.0
    noise : float, optional
        Apply Gaussian noise to particles with width=noise, default=0.0

    Returns
    -------
    box : frued.box.Box
        The box containing the crystal
    positions : np.ndarray, shape=(nx*ny*nz, 3)
        The positions of the particles in the crystal
    """
    fractions = np.array([[.5, .5, 0],
                          [.5, 0, .5],
                          [0, .5, .5],
                          [0, 0, 0]], dtype=np.float32)

    fractions = np.tile(fractions[np.newaxis, np.newaxis, np.newaxis], (nx, ny, nz, 1, 1))
    fractions[..., 0] += np.arange(nx)[:, np.newaxis, np.newaxis, np.newaxis]
    fractions[..., 1] += np.arange(ny)[np.newaxis, :, np.newaxis, np.newaxis]
    fractions[..., 2] += np.arange(nz)[np.newaxis, np.newaxis, :, np.newaxis]
    fractions /= [nx, ny, nz]

    box = 2*scale*np.array([nx, ny, nz], dtype=np.float32)
    positions = ((fractions - .5)*box).reshape((-1, 3))

    if noise != 0:
        positions += np.random.normal(scale=noise, size=positions.shape)

    return freud.box.Box(*box), positions
