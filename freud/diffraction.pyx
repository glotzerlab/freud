import numpy as np
from scipy import interpolate, ndimage
import rowan

# from cme_utils.manip import pbc
# from cme_utils.manip.utilities import rotation_matrix_from_to as cme_rot


class Diffractometer:
    def __init__(
        self,
        grid_size=512,
        zoom=4,
        peak_width=1,
        length_scale=3.905,
        bot=4e-6,
        top=0.7
    ):
        """
        Initialize the diffractometer class
        Parameters
        ----------
        grid_size : int, size of the diffraction grid (default 512)
        zoom : (default 1)
        peak_width : (default 1)
        length_scale : (default 3.905)
        bot : (default 4e-6)
        top : (default 0.7)
        """
        self.N = grid_size
        self.zoom = zoom
        self.peak_width = peak_width
        self.bin_w = 2.0
        self.length_scale = length_scale
        self.bot = bot
        self.top = top

    def load(self, system):
        """
        def load(self, xyz, L):
            Loads the particle positions and box dimensions for diffraction.
            Note: only supports orthorhombic boxes
            Parameters
            ----------
            xyz: np.ndarray (N,3), positions of each particle
            L: iterable object, lengths of box vectors
            self.box = np.array(
                    [[L[0], 0.0, 0.0],
                     [0.0, L[1], 0.0],
                     [0.0, 0.0, L[2]]]
                    )
            self.orig = np.copy(xyz)
            self.orig, self.image =
                pbc.shift_pbc(xyz, np.array([L[0], L[1], L[2]]))
        """
        box, positions = system
        self.box = box.to_matrix()
        self.orig = np.copy(positions)

    def pbc_2d(self, xy, N):
        """
        Reasonably fast periodic boundary conditions in two dimensions.
        Normalizes xy coordinates to the grid size, N.
        Parameters
        ----------
        xy : numpy.ndarray (N,2), cartesian coordinates from [-0.5, 0.5)
        to be mapped to [0, N)
        N : int, grid size
        Returns
        -------
        numpy.ndarray (N,2), particle bins indices
        in the x and y directions.
        """
        xy -= np.rint(xy) - 0.5
        xy *= N
        xy %= N
        return xy.astype(int)

    def bin(self, xy, N):
        """
        Quickly counts intensities for particles on 2D grid.
        Parameters
        ----------
        xy : numpy.ndarray (N,2), array of bin indices
        N : int, grid size
        Returns
        -------
        im : numpy.ndarray (N,N), grid of intensities.
        """
        t = xy.view(np.dtype((np.void, xy.dtype.itemsize * xy.shape[1])))
        _, ids, counts = np.unique(t, return_index=True, return_counts=True)
        unique_xy = xy[ids]
        N = int(N)
        im = np.zeros((N, N))
        for x, c in zip(unique_xy, counts):
            im[x[1], x[0]] = c
        return im

    def calc_proj(self, rot):
        """
        TODO
        Note: orthorhombic boxes only
        Parameters
        ----------
        rot : numpy.ndarray (3,3), rotation matrix
        Returns
        -------
        numpy.ndarray (2,2), inverse shear matrix
        """
        # s = np.dot(rot.T, self.box)  # rotated box vectors
        s = rowan.rotate(rot, self.box)
        xy = np.absolute(s[0, 0] * s[1, 1] - s[0, 1] * s[1, 0])
        zx = np.absolute(s[0, 2] * s[1, 0] - s[0, 0] * s[1, 2])
        yz = np.absolute(s[0, 1] * s[1, 2] - s[0, 2] * s[1, 1])
        if (yz >= xy) and (yz >= zx):
            shear = np.array(
                [[s[0, 1], s[0, 2]], [s[1, 1], s[1, 2]]])
        elif (zx >= xy) and (zx >= yz):
            shear = np.array(
                [[s[0, 2], s[0, 0]], [s[1, 2], s[1, 0]]])
        else:
            shear = np.array(
                [[s[0, 0], s[0, 1]], [s[1, 0], s[1, 1]]])
        s_det = np.linalg.det(shear)
        if s_det == 0:
            print("\nSingular rotation matrix. Bye Bye.")
            return
        self.Lx = np.linalg.norm(shear[:, 0])
        self.Ly = np.linalg.norm(shear[:, 1])
        inv_shear = np.linalg.inv(shear)
        return inv_shear

    def circle_cutout(self, p):
        """
        Find pixel indices in diffraction intensity array outside of the circle
        Note: taken from Diffractometer.prep_sq()
        Parameters
        -------
        p : numpy.ndarray (N,N), diffraction intensity array
        Returns
        -------
        numpy.ndarray (N,), indices of particles outside the circle
        note: N != to N in p.shape
        """
        y, x = np.indices(p.shape)
        rmax = len(x) / 2 - 1
        center = np.array([rmax, rmax])
        # radii, constant for a single zoom
        r = np.hypot(x - center[1], y - center[0]).flatten()
        # array index into p corresponding to r
        i = np.argsort(r.flat)
        # sorted radius indices
        r_sort = r.flat[i]
        return i[r_sort > rmax]

    def scale(self, a):
        """
        Scales up a matrix around middle particle
        Note: Doesn't handle atoms on periodic boundaries perfectly --
        intensity only on one half of boundary.
        Parameters
        -------
        a : numpy.ndarray (N,N), input array
        Returns
        -------
        numpy.ndarray (N,N), scaled array
        """
        ny, nx = np.shape(a)
        y = np.array([list(range(ny))])
        x = np.array([list(range(nx))])
        d = interpolate.RectBivariateSpline(x, y, a, kx=1, ky=1)
        x = np.linspace(0, nx, self.N)
        y = np.linspace(0, ny, self.N)
        d = d(x, y)
        return d

    def shear_back(self, img, inv_shear):
        """
        TODO
        Parameters
        ----------
        img : numpy.ndarray (N,N), array of diffraction intensities
        inv_shear : numpy.ndarray (2,2), inverse shear matrix
        Returns
        -------
        numpy.ndarray (N,N), sheared array of diffraction intensities
        """
        roll = img.shape[0] / 2 - 1
        ss = np.max(self.box) * inv_shear
        A1 = np.array([[1, 0, -roll],
                      [0, 1, -roll],
                      [0, 0, 1]])

        A2 = np.array([[ss[1, 0], ss[0, 0], roll],
                      [ss[1, 1], ss[0, 1], roll],
                      [0, 0, 1]])

        A3 = np.linalg.inv(np.dot(A2, A1))
        A4 = A3[0:2, 0:2]
        A5 = A3[0:2, 2]
        img = ndimage.interpolation.affine_transform(img, A4, A5,
                                                     mode="constant")
        return img

    def compute(self, rot, cutout=True):
        """
        2D FFT to get diffraction pattern from intensity matrix.
        Parameters
        ----------
        rot: numpy.ndarray (3, 3), rotation matrix
        cutout: bool, return diffraction pattern with circle cutout
        (default True)
        Returns
        -------
        numpy.ndarray (N,N), diffraction pattern
        """
        N = self.N / self.zoom
        inv_shear = self.calc_proj(rot)
        xy = np.copy(rowan.rotate(rot, self.orig)[:, 0:2])
        xy = np.dot(xy, inv_shear.T)
        xy = self.pbc_2d(xy, N)
        im = self.bin(xy, N)

        dp = np.fft.fft2(im)
        dp = ndimage.fourier.fourier_gaussian(dp, self.peak_width / self.zoom)
        dp = np.fft.fftshift(dp)
        dp = np.absolute(dp)
        dp *= dp

        dp = self.scale(dp)
        dp = self.shear_back(dp, inv_shear)
        dp /= dp.max()
        dp[dp < self.bot] = self.bot
        dp[dp > self.top] = self.top
        dp = np.log(dp)

        """
        NOTE: cut into a circle, not sure if needed-YJ
        """
        # if not cutout:
        #     return dp

        # idbig = self.circle_cutout(dp)
        # dp[np.unravel_index(idbig, (self.N, self.N))] = np.log(self.bot)
        return dp


def vector_projection(u, v):
    """
    Projection of u onto v
    Parameters
    ----------
    u,v : numpy.ndarray (3,), vectors
    Returns
    -------
    numpy.ndarray (3,), projection of u onto v
    """
    return v * np.dot(u, v)/np.linalg.norm(v)


def unit_vector(vector):
    """
    Returns the unit vector of the vector.
    """
    return vector / np.linalg.norm(vector)


def get_angle(u, v):
    """
    Find angle between u and v
    Parameters
    ----------
    u,v : numpy.ndarray (3,), vectors
    Returns
    -------
    float, angle between u and v in radians
    """
    u = unit_vector(u)
    v = unit_vector(v)
    angle = np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))
    if angle != angle:
        # Catches nan values
        return 0.0
    return angle


# def camera_to_rot(camera):
#     """
#     Given a fresnel camera object, compute the rotation matrix
#     Parameters
#     ----------
#     camera : fresnel.camera, camera in fresnel scene
#     Returns
#     -------
#     numpy.ndarray (3,3), rotation matrix
#     """
#     pos = camera.position
#     look_at = camera.look_at

#     cam_vec = np.array(pos)-np.array(look_at)

#     return cme_rot(cam_vec, np.array([0,0,1]))

# def rot_mat(alpha, beta, gamma):
#     """
#     Given angles alpha, beta, and gamma, compute the rotation matrix
#     Parameters
#     ----------
#     alpha, beta, gamma : float, angles about the x, y, and z axes in radians
#     Returns
#     -------
#     numpy.ndarray (3,3), rotation matrix
#     """
#     Rx = np.array([
#         [1, 0, 0],
#         [0, np.cos(alpha), -np.sin(alpha)],
#         [0, np.sin(alpha), np.cos(alpha)]
#     ])
#     Ry = np.array([
#         [np.cos(beta), 0, np.sin(beta)],
#         [0, 1, 0],
#         [-np.sin(beta), 0, np.cos(beta)]
#     ])
#     Rz = np.array([
#         [np.cos(gamma), -np.sin(gamma), 0],
#         [np.sin(gamma), np.cos(gamma), 0],
#         [0, 0, 1]
#     ])
#     return np.dot(np.dot(Rx,Ry),Rz)
