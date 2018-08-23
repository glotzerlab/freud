import numpy as np
import numpy.testing as npt
from freud import box, bond
from freud.errors import FreudDeprecationWarning
import warnings
import unittest


class TestBond(unittest.TestCase):
    def setUp(self):
        # We ignore warnings for test_2_dimensional
        warnings.simplefilter("ignore", category=FreudDeprecationWarning)

    def test_correct_bond(self):
        # generate the bonding map
        nr = 10
        nt1 = 100
        nt2 = 50
        testArray = np.zeros(shape=(nr, nt2, nt1), dtype=np.uint32)
        rmax = 3.0
        dr = rmax / float(nr)
        dt2 = 2.0 * np.pi / float(nt2)
        dt1 = 2.0 * np.pi / float(nt1)

        # make sure the radius for each bin is generated correctly
        posList = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                           dtype=np.float32)
        anglist = np.zeros(shape=(2), dtype=np.float32)

        # calculate the bin
        deltaX = posList[1, 0] - posList[0, 0]
        deltaY = posList[1, 1] - posList[0, 1]
        delta = np.array([deltaX, deltaY], dtype=np.float32)
        r = np.sqrt(np.dot(delta, delta))
        theta1 = anglist[0] - np.arctan2(deltaY, deltaX)
        theta2 = anglist[1] - np.arctan2(-deltaY, -deltaX)
        theta1 = theta1 if (theta1 > 0) else theta1 + 2.0*np.pi
        theta1 = theta1 if (theta1 < 2.0*np.pi) else theta1 - 2.0*np.pi
        theta2 = theta2 if (theta2 > 0) else theta2 + 2.0*np.pi
        theta2 = theta2 if (theta2 < 2.0*np.pi) else theta2 - 2.0*np.pi
        binR = int(np.floor(r / dr))
        binT1 = int(np.floor(theta1 / dt1))
        binT2 = int(np.floor(theta2 / dt2))
        testArray[binR, binT2, binT1] = 5
        deltaX = posList[0, 0] - posList[1, 0]
        deltaY = posList[0, 1] - posList[1, 1]
        delta = np.array([deltaX, deltaY], dtype=np.float32)
        r = np.sqrt(np.dot(delta, delta))
        theta1 = anglist[0] - np.arctan2(deltaY, deltaX)
        theta2 = anglist[1] - np.arctan2(-deltaY, -deltaX)
        theta1 = theta1 if (theta1 > 0) else theta1 + 2.0*np.pi
        theta1 = theta1 if (theta1 < 2.0*np.pi) else theta1 - 2.0*np.pi
        theta2 = theta2 if (theta2 > 0) else theta2 + 2.0*np.pi
        theta2 = theta2 if (theta2 < 2.0*np.pi) else theta2 - 2.0*np.pi
        binR = int(np.floor(r / dr))
        binT1 = int(np.floor(theta1 / dt1))
        binT2 = int(np.floor(theta2 / dt2))
        testArray[binR, binT2, binT1] = 5

        # create object
        bondList = np.array([0, 5], dtype=np.uint32)
        EB = bond.BondingR12(rmax, testArray, bondList)

        # create the box
        f_box = box.Box(Lx=5.0*rmax, Ly=5.0*rmax, is2D=True)

        # run the computation
        EB.compute(f_box, posList, anglist, posList, anglist)

        # check to make sure that the point is in the correct bin

        bonds = EB.bonds

        npt.assert_equal(bonds[0, 1], 1)
        npt.assert_equal(bonds[1, 1], 0)

    def test_mapping(self):
        # generate the bonding map
        nr = 10
        nt1 = 100
        nt2 = 50
        testArray = np.zeros(shape=(nr, nt2, nt1), dtype=np.uint32)
        rmax = 3.0

        # create object
        bondList = np.array([0, 3, 4, 5], dtype=np.uint32)
        EB = bond.BondingR12(rmax, testArray, bondList)

        bond_map = EB.list_map

        npt.assert_equal(bond_map[3], 1)

    def test_rev_mapping(self):
        # generate the bonding map
        nr = 10
        nt1 = 100
        nt2 = 50
        testArray = np.zeros(shape=(nr, nt2, nt1), dtype=np.uint32)
        rmax = 3.0

        # create object
        bondList = np.array([0, 3, 4, 5], dtype=np.uint32)
        EB = bond.BondingR12(rmax, testArray, bondList)

        bond_map = EB.rev_list_map

        npt.assert_equal(bond_map[1], bondList[1])


if __name__ == '__main__':
    unittest.main()
