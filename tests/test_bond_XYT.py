import numpy as np
import numpy.testing as npt
from freud import box, bond
from freud.errors import FreudDeprecationWarning
import warnings
import unittest


class TestBond(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=FreudDeprecationWarning)

    def test_correct_bond(self):
        # generate the bonding map
        nx = 60
        ny = 80
        nt = 50
        testArray = np.zeros(shape=(nt, ny, nx), dtype=np.uint32)
        xmax = 3.0
        ymax = 5.0
        tmax = 2.0*np.pi
        rmax = np.sqrt(xmax**2 + ymax**2)
        dx = 2.0 * xmax / float(nx)
        dy = 2.0 * ymax / float(ny)
        dt = tmax / float(nt)

        # make sure the radius for each bin is generated correctly
        posList = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                           dtype=np.float32)
        anglist = np.zeros(shape=(2), dtype=np.float32)

        # calculate the bin
        deltaX = posList[1, 0] - posList[0, 0]
        deltaY = posList[1, 1] - posList[0, 1]
        theta = anglist[1] - np.arctan2(-deltaY, -deltaX)
        theta = theta if (theta > 0) else theta + 2.0*np.pi
        theta = theta if (theta < 2.0*np.pi) else theta - 2.0*np.pi
        x = deltaX + xmax
        y = deltaY + ymax
        binX = int(np.floor(x / dx))
        binY = int(np.floor(y / dy))
        binT = int(np.floor(theta / dt))
        testArray[binT, binY, binX] = 5

        deltaX = posList[0, 0] - posList[1, 0]
        deltaY = posList[0, 1] - posList[1, 1]
        theta = anglist[1] - np.arctan2(-deltaY, -deltaX)
        theta = theta if (theta > 0) else theta + 2.0*np.pi
        theta = theta if (theta < 2.0*np.pi) else theta - 2.0*np.pi
        x = deltaX + xmax
        y = deltaY + ymax
        binX = int(np.floor(x / dx))
        binY = int(np.floor(y / dy))
        binT = int(np.floor(theta / dt))
        testArray[binT, binY, binX] = 5

        # create object
        bondList = np.array([0, 5], dtype=np.uint32)
        EB = bond.BondingXYT(xmax, ymax, testArray, bondList)

        # create the box
        f_box = box.Box.square(5.0*rmax)

        # run the computation
        EB.compute(f_box, posList, anglist, posList, anglist)

        # check to make sure that the point is in the correct bin

        bonds = EB.bonds

        npt.assert_equal(bonds[0, 1], 1)
        npt.assert_equal(bonds[1, 1], 0)

        # Ensure appropriate error is raised
        f_box = box.Box.cube(5.0*rmax)
        with self.assertRaises(ValueError):
            EB.compute(f_box, posList, anglist, posList, anglist)

    def test_mapping(self):
        # generate the bonding map
        nx = 60
        ny = 80
        nt = 50
        testArray = np.zeros(shape=(nt, ny, nx), dtype=np.uint32)
        xmax = 3.0
        ymax = 5.0

        # create object
        bondList = np.array([0, 3, 4, 5], dtype=np.uint32)
        EB = bond.BondingXYT(xmax, ymax, testArray, bondList)

        bond_map = EB.list_map

        npt.assert_equal(bond_map[3], 1)

    def test_rev_mapping(self):
        # generate the bonding map
        nx = 60
        ny = 80
        nt = 50
        testArray = np.zeros(shape=(nt, ny, nx), dtype=np.uint32)
        xmax = 3.0
        ymax = 5.0

        # create object
        bondList = np.array([0, 3, 4, 5], dtype=np.uint32)
        EB = bond.BondingXYT(xmax, ymax, testArray, bondList)

        bond_map = EB.rev_list_map

        npt.assert_equal(bond_map[1], bondList[1])


if __name__ == '__main__':
    unittest.main()
