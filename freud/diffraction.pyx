import numpy as np
from scipy import interpolate, ndimage
import rowan

from freud.util import _Compute


class Diffraction(_Compute):
    def __init__(self, grid_size=512, zoom=4, peak_width=1,
                 length_scale=3.905, bot=4e-6, top=0.7):
        """
        Initialize the diffraction class
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

    def compute(self, system, rot, cutout=True):
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
        self.box, self.positions = system
        self.box = self.box.to_matrix()
        N = self.N / self.zoom
        inv_shear = self.calc_proj(rot)
        xy = np.copy(rowan.rotate(rot, self.orig)[:, 0:2])
        xy = np.dot(xy, inv_shear.T)
        xy = self.pbc_2d(xy, N)
        im = self.bin(xy, N)

        self.dp = np.fft.fft2(im)
        self.dp = ndimage.fourier.fourier_gaussian(self.dp,
                                                   self.peak_width / self.zoom)
        self.dp = np.fft.fftshift(self.dp)
        self.dp = np.absolute(self.dp)
        self.dp *= self.dp

        self.dp = self.scale(self.dp)
        self.dp = self.shear_back(self.dp, inv_shear)
        self.dp /= self.dp.max()
        self.dp[self.dp < self.bot] = self.bot
        self.dp[self.dp > self.top] = self.top
        self.dp = np.log10(self.dp)

        """
        NOTE: cut into a circle, not sure if needed-YJ
        """
        # if not cutout:
        #     return dp

        # idbig = self.circle_cutout(dp)
        # dp[np.unravel_index(idbig, (self.N, self.N))] = np.log(self.bot)
        return self.dp

    @_Compute._computed_property
    def diffraction(self):
        return self.dp

    def __repr__(self):
        return f"freud.diffraction.{type(self).__name__}, (N={self.N}, \
                 zoom={self.zoom}, peak_width={self.peak_width}, \
                 length_scale={self.length_scale}, bot={self.bot}, \
                 top={self.top})"

    def plot(self, ax=None):
        """Plot Diffraction Pattern.

        Args:
            ax (:class:`matplotlib.axes.Axes`, optional): Axis to plot on. If
                :code:`None`, make a new figure and axis.
                (Default value = :code:`None`)

        Returns:
            (:class:`matplotlib.axes.Axes`): Axis with the plot.
        """
        import freud.plot
        return freud.plot.diffraction_plot(self.dp)

    def _repr_png_(self):
        try:
            import freud.plot
            return freud.plot._ax_to_bytes(self.plot())
        except (AttributeError, ImportError):
            return None
