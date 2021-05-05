import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pytest

import freud

matplotlib.use("agg")


class TestGaussianDensity:
    def test_random_point_with_cell_list(self):
        fftpack = pytest.importorskip("scipy.fftpack")
        fft = fftpack.fft
        fftshift = fftpack.fftshift

        width = 20
        r_max = 10.0
        sigma = 0.1
        num_points = 10000
        box_size = r_max * 3.1
        box, points = freud.data.make_random_system(box_size, num_points, is2D=True)
        for w in (width, (width, width), [width, width]):
            gd = freud.density.GaussianDensity(w, r_max, sigma)

            # Test access
            with pytest.raises(AttributeError):
                gd.box
            with pytest.raises(AttributeError):
                gd.density

            gd.compute((box, points))

            # Test access
            gd.box
            gd.density

            # Verify the output dimensions are correct
            assert gd.density.shape == (width, width)
            assert np.prod(gd.density.shape) == np.prod(gd.width)

            myDiff = gd.density
            myFFT = fft(fft(myDiff[:, :], axis=1), axis=0)
            myDiff = (myFFT * np.conj(myFFT)).real
            myDiff = fftshift(myDiff)[:, :]
            npt.assert_equal(
                np.where(myDiff == np.max(myDiff)),
                (np.array([width // 2]), np.array([width // 2])),
            )

    def test_change_box_dimension(self):
        width = 20
        r_max = 9.9
        sigma = 0.01
        num_points = 100
        box_size = r_max * 3.1

        # test that a 3D system computed after computing a 2D system will fail
        box, points = freud.data.make_random_system(box_size, num_points, is2D=True)
        gd = freud.density.GaussianDensity(width, r_max, sigma)
        gd.compute((box, points))

        test_box, test_points = freud.data.make_random_system(
            box_size, num_points, is2D=False
        )
        with pytest.raises(ValueError):
            gd.compute((test_box, test_points))

        # test that a 2D system computed after computing a 3D system will fail
        box, points = freud.data.make_random_system(box_size, num_points, is2D=False)
        gd = freud.density.GaussianDensity(width, r_max, sigma)
        gd.compute((box, points))

        test_box, test_points = freud.data.make_random_system(
            box_size, num_points, is2D=True
        )
        with pytest.raises(ValueError):
            gd.compute((test_box, test_points))

    def test_sum_2d(self):
        # Ensure that each point's Gaussian sums to 1
        width = 20
        r_max = 9.9
        sigma = 2
        box_size = width
        gd = freud.density.GaussianDensity(width, r_max, sigma)
        for num_points in [1, 10, 100]:
            box, points = freud.data.make_random_system(box_size, num_points, is2D=True)
            gd.compute(system=(box, points))
            # This has discretization error as well as single-precision error
            assert np.isclose(np.sum(gd.density), num_points, rtol=1e-4)

    def test_sum_3d(self):
        # Ensure that each point's Gaussian sums to 1
        width = 20
        r_max = 9.9
        sigma = 2
        box_size = width
        gd = freud.density.GaussianDensity(width, r_max, sigma)
        for num_points in [1, 10, 100]:
            box, points = freud.data.make_random_system(
                box_size, num_points, is2D=False
            )
            gd.compute(system=(box, points))
            # This has discretization error as well as single-precision error
            assert np.isclose(np.sum(gd.density), num_points, rtol=1e-4)

    def test_sum_values_2d(self):
        # Ensure that the Gaussian convolution sums to the sum of the values
        width = 20
        r_max = 9.9
        sigma = 2
        box_size = width
        gd = freud.density.GaussianDensity(width, r_max, sigma)
        for num_points in [1, 10, 100]:
            system = freud.data.make_random_system(box_size, num_points, is2D=True)
            values = np.random.rand(num_points)
            gd.compute(system, values)
            # This has discretization error as well as single-precision error
            assert np.isclose(np.sum(gd.density), np.sum(values), rtol=1e-4)

    def test_sum_values_3d(self):
        # Ensure that the Gaussian convolution sums to the sum of the values
        width = 20
        r_max = 9.9
        sigma = 2
        box_size = width
        gd = freud.density.GaussianDensity(width, r_max, sigma)
        for num_points in [1, 10, 100]:
            system = freud.data.make_random_system(box_size, num_points, is2D=False)
            values = np.random.rand(num_points)
            gd.compute(system, values)
            # This has discretization error as well as single-precision error
            assert np.isclose(np.sum(gd.density), np.sum(values), rtol=1e-4)

    def test_repr(self):
        gd = freud.density.GaussianDensity(100, 10.0, 0.1)
        assert str(gd) == str(eval(repr(gd)))

        # Use both signatures
        gd3 = freud.density.GaussianDensity((98, 99, 100), 10.0, 0.1)
        assert str(gd3) == str(eval(repr(gd3)))

    def test_repr_png(self):
        width = 20
        r_max = 2.0
        sigma = 0.01
        num_points = 100
        box_size = r_max * 3.1
        box, points = freud.data.make_random_system(box_size, num_points, is2D=True)
        gd = freud.density.GaussianDensity(width, r_max, sigma)

        with pytest.raises(AttributeError):
            gd.plot()
        assert gd._repr_png_() is None

        gd.compute((box, points))
        gd.plot()

        gd = freud.density.GaussianDensity(width, r_max, sigma)
        test_box = freud.box.Box.cube(box_size)
        gd.compute((test_box, points))
        gd.plot()
        assert gd._repr_png_() is None
        plt.close("all")
