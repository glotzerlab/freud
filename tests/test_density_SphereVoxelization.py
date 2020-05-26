import numpy as np
import freud
import unittest
from SphereVoxelization_fft import compute_3d, compute_2d


class TestSphereVoxelization(unittest.TestCase):
    def test_random_points_2d(self):
        width = 100
        r_max = 10.0
        num_points = 10
        box_size = r_max*10
        box, points = freud.data.make_random_system(
            box_size, num_points, is2D=True)
        for w in (width, (width, width), [width, width]):
            vox = freud.density.SphereVoxelization(w, r_max)

            # Test access
            with self.assertRaises(AttributeError):
                vox.box
            with self.assertRaises(AttributeError):
                vox.voxels

            vox.compute(system=(box, points))

            # Test access
            vox.box
            vox.voxels

            # Verify the output dimensions are correct
            self.assertEqual(vox.voxels.shape, (width, width))
            self.assertEqual(np.prod(vox.voxels.shape), np.prod(vox.width))

            # Verify the calculation is correct
            # here we assert that the calculations (from two different methods)
            # are the same up to rounding error
            fft_vox = compute_2d(box_size, width, points, r_max)
            num_same = len(np.where(np.isclose(vox.voxels - fft_vox,
                                               np.zeros(fft_vox.shape)))[0])
            total_num = np.prod(fft_vox.shape)
            self.assertGreater(num_same / total_num, .95)

            # Verify that the voxels are all 1's and 0's
            num_zeros = len(np.where(np.isclose(
                vox.voxels, np.zeros(vox.voxels.shape)))[0])
            num_ones = len(np.where(np.isclose(
                vox.voxels, np.ones(vox.voxels.shape)))[0])
            self.assertGreater(num_zeros, 0)
            self.assertGreater(num_ones, 0)
            self.assertEqual(num_zeros + num_ones, np.prod(vox.voxels.shape))

    def test_random_points_3d(self):
        width = 100
        r_max = 10.0
        num_points = 10
        box_size = r_max*10
        box, points = freud.data.make_random_system(
            box_size, num_points, is2D=False)
        for w in (width, (width, width, width), [width, width, width]):
            vox = freud.density.SphereVoxelization(w, r_max)

            # Test access
            with self.assertRaises(AttributeError):
                vox.box
            with self.assertRaises(AttributeError):
                vox.voxels

            vox.compute(system=(box, points))

            # Test access
            vox.box
            vox.voxels

            # Verify the output dimensions are correct
            self.assertEqual(vox.voxels.shape, (width, width, width))

            # Verify the calculation is correct
            # here we assert that the calculations (from two different methods)
            # are the same up to rounding error
            fft_vox = compute_3d(box_size, width, points, r_max)
            num_same = len(np.where(np.isclose(vox.voxels - fft_vox,
                                               np.zeros(fft_vox.shape)))[0])
            total_num = np.prod(fft_vox.shape)
            self.assertGreater(num_same / total_num, .95)

            # Verify that the voxels are all 1's and 0's
            num_zeros = len(np.where(np.isclose(
                vox.voxels, np.zeros(vox.voxels.shape)))[0])
            num_ones = len(np.where(np.isclose(
                vox.voxels, np.ones(vox.voxels.shape)))[0])
            self.assertGreater(num_zeros, 0)
            self.assertGreater(num_ones, 0)
            self.assertEqual(num_zeros + num_ones, np.prod(vox.voxels.shape))

    def test_change_box_dimension(self):
        width = 100
        r_max = 10.0
        num_points = 100
        box_size = r_max*3.1

        # test that computing a 3D system after computing a 2D system will fail
        box, points = freud.data.make_random_system(
            box_size, num_points, is2D=True)
        vox = freud.density.SphereVoxelization(width, r_max)
        vox.compute(system=(box, points))

        test_box, test_points = freud.data.make_random_system(
            box_size, num_points, is2D=False)
        with self.assertRaises(ValueError):
            vox.compute((test_box, test_points))

        # test that computing a 2D system after computing a 3D system will fail
        box, points = freud.data.make_random_system(
            box_size, num_points, is2D=False)
        vox = freud.density.SphereVoxelization(width, r_max)
        vox.compute(system=(box, points))

        test_box, test_points = freud.data.make_random_system(
            box_size, num_points, is2D=True)
        with self.assertRaises(ValueError):
            vox.compute((test_box, test_points))

    def test_repr(self):
        vox = freud.density.SphereVoxelization(100, 10.0)
        self.assertEqual(str(vox), str(eval(repr(vox))))

        # Use both signatures
        vox3 = freud.density.SphereVoxelization((98, 99, 100), 10.0)
        self.assertEqual(str(vox3), str(eval(repr(vox3))))

    def test_repr_png(self):
        width = 100
        r_max = 10.0
        num_points = 100
        box_size = r_max*3.1
        box, points = freud.data.make_random_system(
            box_size, num_points, is2D=True)
        vox = freud.density.SphereVoxelization(width, r_max)

        with self.assertRaises(AttributeError):
            vox.plot()
        self.assertEqual(vox._repr_png_(), None)

        vox.compute((box, points))
        vox.plot()

        vox = freud.density.SphereVoxelization(width, r_max)
        test_box = freud.box.Box.cube(box_size)
        vox.compute((test_box, points))
        vox.plot()
        self.assertEqual(vox._repr_png_(), None)


if __name__ == '__main__':
    unittest.main()
