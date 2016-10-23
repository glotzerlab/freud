import numpy
import numpy as np
import numpy.testing as npt
from freud import box, bond, parallel
import unittest

class TestBond(unittest.TestCase):
    def test_correct_bond(self):
        # generate the bonding map
        nx = 60
        ny = 80
        testArray = np.zeros(shape=(ny, nx), dtype=np.uint32)
        xmax = 3.0
        ymax = 5.0
        rmax = np.sqrt(xmax**2 + ymax**2)
        dx = 2.0 * xmax / float(nx)
        dy = 2.0 * ymax / float(ny)

        # make sure the radius for each bin is generated correctly
        posList = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)
        anglist = np.zeros(shape=(2), dtype=np.float32)

        # calculate the bin
        deltaX = posList[0][0] - posList[1][0]
        deltaY = posList[0][1] - posList[1][1]
        x = deltaX + xmax
        y = deltaY + ymax
        binX = int(numpy.floor(x / dx))
        binY = int(numpy.floor(y / dy))
        testArray[binY,binX] = 5
        deltaX = posList[1][0] - posList[0][0]
        deltaY = posList[1][1] - posList[0][1]
        x = deltaX + xmax
        y = deltaY + ymax
        binX = int(numpy.floor(x / dx))
        binY = int(numpy.floor(y / dy))
        testArray[binY,binX] = 5

        # create object
        bondList = np.array([0, 5], dtype=np.uint32)
        EB = bond.BondingXY2D(xmax, ymax, testArray, bondList)

        # create the box
        f_box = box.Box.square(5.0*rmax)

        # run the computation
        EB.compute(f_box, posList, anglist, posList, anglist)

        # check to make sure that the point is in the correct bin

        bonds = EB.getBonds()

        npt.assert_equal(bonds[0,1], 1)
        npt.assert_equal(bonds[1,1], 0)

    def test_mapping(self):
        # generate the bonding map
        nx = 60
        ny = 80
        testArray = np.zeros(shape=(ny, nx), dtype=np.uint32)
        xmax = 3.0
        ymax = 5.0
        rmax = np.sqrt(xmax**2 + ymax**2)
        dx = 2.0 * xmax / float(nx)
        dy = 2.0 * ymax / float(ny)

        # create object
        bondList = np.array([0, 3, 4, 5], dtype=np.uint32)
        EB = bond.BondingXY2D(xmax, ymax, testArray, bondList)

        bond_map = EB.getListMap()

        npt.assert_equal(bond_map[3], 1)

    def test_rev_mapping(self):
        # generate the bonding map
        nx = 60
        ny = 80
        testArray = np.zeros(shape=(ny, nx), dtype=np.uint32)
        xmax = 3.0
        ymax = 5.0
        rmax = np.sqrt(xmax**2 + ymax**2)
        dx = 2.0 * xmax / float(nx)
        dy = 2.0 * ymax / float(ny)

        # create object
        bondList = np.array([0, 3, 4, 5], dtype=np.uint32)
        EB = bond.BondingXY2D(xmax, ymax, testArray, bondList)

        bond_map = EB.getRevListMap()

        npt.assert_equal(bond_map[1], bondList[1])

if __name__ == '__main__':
    unittest.main()
