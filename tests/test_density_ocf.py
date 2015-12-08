import numpy as np
import numpy.testing as npt
from freud import trajectory, density
import unittest

class TestR(unittest.TestCase):
    def test_generateR(self):
        rmax = 51.23
        dr = 0.1
        nbins = int(rmax / dr)

        # make sure the radius for each bin is generated correctly
        r_list = np.zeros(nbins, dtype=np.float32)
        for i in range(nbins):
            r1 = i * dr
            r2 = r1 + dr
            r_list[i] = 2.0/3.0 * (r2**3.0 - r1**3.0) / (r2**2.0 - r1**2.0)

        ocf = density.ComplexCF(rmax, dr)

        npt.assert_almost_equal(ocf.getR(), r_list, decimal=3)

class TestOCF(unittest.TestCase):
    def test_random_point_with_cell_list(self):
        rmax = 10.0
        dr = 1.0
        num_points = 10000
        box_size = rmax*3.1
        points = np.random.random_sample((num_points,3)).astype(np.float32)*box_size - box_size/2
        ang = np.random.random_sample((num_points)).astype(np.float32)*np.pi*2.0
        comp = np.cos(ang) + 1j * np.sin(ang)
        conj = np.cos(ang) - 1j * np.sin(ang)
        ocf = density.ComplexCF(rmax, dr)
        ocf.accumulate(trajectory.Box(box_size),points, comp, points, conj)

        correct = np.zeros(int(rmax/dr), dtype=np.complex64)
        absolute_tolerance = 0.1
        # first bin is bad
        npt.assert_allclose(ocf.getRDF(), correct, atol=absolute_tolerance)

    def test_random_point_without_cell_list(self):
        rmax = 10.0
        dr = 1.0
        num_points = 10000
        box_size = rmax*2
        points = np.random.random_sample((num_points,3)).astype(np.float32)*box_size - box_size/2
        ang = np.random.random_sample((num_points)).astype(np.float32)*np.pi*2.0
        comp = np.cos(ang) + 1j * np.sin(ang)
        conj = np.cos(ang) - 1j * np.sin(ang)
        ocf = density.ComplexCF(rmax, dr)
        ocf.accumulate(trajectory.Box(box_size),points, comp, points, conj)

        correct = np.zeros(int(rmax/dr), dtype=np.complex64)
        absolute_tolerance = 0.1
        # first bin is bad
        npt.assert_allclose(ocf.getRDF(), correct, atol=absolute_tolerance)

    def test_value_point_with_cell_list(self):
        rmax = 10.0
        dr = 1.0
        num_points = 10000
        box_size = rmax*3.1
        points = np.random.random_sample((num_points,3)).astype(np.float32)*box_size - box_size/2
        ang = np.zeros(int(num_points), dtype=np.float32)
        comp = np.cos(ang) + 1j * np.sin(ang)
        conj = np.cos(ang) - 1j * np.sin(ang)
        ocf = density.ComplexCF(rmax, dr)
        ocf.accumulate(trajectory.Box(box_size),points, comp, points, conj)

        correct = np.ones(int(rmax/dr), dtype=np.float32) + 1j * np.zeros(int(rmax/dr), dtype=np.float32)
        absolute_tolerance = 0.1
        npt.assert_allclose(ocf.getRDF(), correct, atol=absolute_tolerance)

    def test_value_point_without_cell_list(self):
        rmax = 10.0
        dr = 1.0
        num_points = 10000
        box_size = rmax*2
        points = np.random.random_sample((num_points,3)).astype(np.float32)*box_size - box_size/2
        ang = np.zeros(int(num_points), dtype=np.float32)
        comp = np.cos(ang) + 1j * np.sin(ang)
        conj = np.cos(ang) - 1j * np.sin(ang)
        ocf = density.ComplexCF(rmax, dr)
        ocf.accumulate(trajectory.Box(box_size),points, comp, points, conj)

        correct = np.ones(int(rmax/dr), dtype=np.float32) + 1j * np.zeros(int(rmax/dr), dtype=np.float32)
        absolute_tolerance = 0.1
        npt.assert_allclose(ocf.getRDF(), correct, atol=absolute_tolerance)

if __name__ == '__main__':
    unittest.main()
