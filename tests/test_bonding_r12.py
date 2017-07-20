import numpy
import numpy as np
import numpy.testing as npt
from freud import box, bond, parallel
import unittest

class TestBond(unittest.TestCase):
    def test_correct_bond(self):
        # generate the bonding map
        nr = 10
        nt1 = 100
        nt2 = 50
        testArray = np.zeros(shape=(nr, nt1, nt2), dtype=np.uint32)
        rmax = 3.0
        dr = rmax / float(nr)
        dt2 = 2.0 * np.pi / float(nt2)
        dt1 = 2.0 * np.pi / float(nt1)

        # make sure the radius for each bin is generated correctly
        # posList = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)
        posList = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)
        # angList = np.zeros(shape=(2), dtype=np.float32)
        angList = np.zeros(shape=(3), dtype=np.float32)

        # calculate the bin
        # particle 1
        # calculate interparticle vector
        deltaX = posList[1,0] - posList[0,0]
        deltaY = posList[1,1] - posList[0,1]
        delta = np.array([deltaX, deltaY], dtype=np.float32)
        # find length of the vector
        r = np.sqrt(np.dot(delta, delta))
        # find angles
        theta1 = angList[0] - np.arctan2(deltaY, deltaX)
        theta2 = angList[1] - np.arctan2(-deltaY, -deltaX)
        # deal with 0, 2pi range
        theta1 = theta1 if (theta1 > 0) else theta1 + 2.0*np.pi
        theta1 = theta1 if (theta1 < 2.0*np.pi) else theta1 - 2.0*np.pi
        theta2 = theta2 if (theta2 > 0) else theta2 + 2.0*np.pi
        theta2 = theta2 if (theta2 < 2.0*np.pi) else theta2 - 2.0*np.pi
        print("t1 = {}".format(theta1))
        print("t2 = {}".format(theta2))
        # bin the values
        binR = int(numpy.floor(r / dr))
        binT1 = int(numpy.floor(theta1 / dt1))
        binT2 = int(numpy.floor(theta2 / dt2))
        print("bin: {}, {}, {}".format(binR, binT1, binT2))
        # insert values into the array
        testArray[binR,binT1,binT2] = 5
        deltaX = posList[0,0] - posList[1,0]
        deltaY = posList[0,1] - posList[1,1]
        delta = np.array([deltaX, deltaY], dtype=np.float32)
        r = np.sqrt(np.dot(delta, delta))
        theta1 = angList[0] - np.arctan2(deltaY, deltaX)
        theta2 = angList[1] - np.arctan2(-deltaY, -deltaX)
        theta1 = theta1 if (theta1 > 0) else theta1 + 2.0*np.pi
        theta1 = theta1 if (theta1 < 2.0*np.pi) else theta1 - 2.0*np.pi
        theta2 = theta2 if (theta2 > 0) else theta2 + 2.0*np.pi
        theta2 = theta2 if (theta2 < 2.0*np.pi) else theta2 - 2.0*np.pi
        binR = int(numpy.floor(r / dr))
        binT1 = int(numpy.floor(theta1 / dt1))
        binT2 = int(numpy.floor(theta2 / dt2))
        testArray[binR,binT1,binT2] = 5
        testArray[:,:,:] = 5

        # create object
        bondList = np.array([0, 5], dtype=np.uint32)
        EB = bond.BondingR12(rmax, testArray, bondList)

        # create the box
        f_box = box.Box(Lx=5.0*rmax, Ly=5.0*rmax, is2D=True)

        # run the computation
        EB.initialize(f_box, posList, angList, posList, angList)
        EB.compute(f_box, posList, angList, posList, angList)
        EB.compute(f_box, posList, angList, posList, angList)
        EB.compute(f_box, posList, angList, posList, angList)
        # EB.compute(f_box, posList, angList, posList, angList)

        # check to make sure that the point is in the correct bin

        bonds = EB.getBonds()
        print(bonds)
        listMap = EB.getListMap()
        print(listMap)
        revMap = EB.getRevListMap()
        print(revMap)
        # bonds = EB.getBondLifetimes()

        # check that particle 0 is bound to particle 1
        npt.assert_equal(bonds[0][0][0], 1)
        # check that particle 1 is bound to particle 0
        npt.assert_equal(bonds[1][0][0], 0)
        # npt.assert_equal(bonds[1,1], 0)

    def test_mapping(self):
        # generate the bonding map
        nr = 10
        nt1 = 100
        nt2 = 50
        testArray = np.zeros(shape=(nr, nt2, nt1), dtype=np.uint32)
        rmax = 3.0
        dr = rmax / float(nr)
        dt2 = 2.0 * np.pi / float(nt2)
        dt1 = 2.0 * np.pi / float(nt1)

        # create object
        bondList = np.array([0, 3, 4, 5], dtype=np.uint32)
        EB = bond.BondingR12(rmax, testArray, bondList)

        bond_map = EB.getListMap()

        npt.assert_equal(bond_map[3], 1)

    def test_rev_mapping(self):
        # generate the bonding map
        nr = 10
        nt1 = 100
        nt2 = 50
        testArray = np.zeros(shape=(nr, nt2, nt1), dtype=np.uint32)
        rmax = 3.0
        dr = rmax / float(nr)
        dt2 = 2.0 * np.pi / float(nt2)
        dt1 = 2.0 * np.pi / float(nt1)

        # create object
        bondList = np.array([0, 3, 4, 5], dtype=np.uint32)
        EB = bond.BondingR12(rmax, testArray, bondList)

        bond_map = EB.getRevListMap()

        npt.assert_equal(bond_map[1], bondList[1])

if __name__ == '__main__':
    unittest.main()
