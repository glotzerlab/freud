import numpy
import numpy as np
import numpy.testing as npt
from freud import trajectory, order, parallel
import unittest

class TestBond(unittest.TestCase):
    def test_correct_bond(self):
        # generate the bonding map
        nx = ny = 10
        testArray = np.zeros(shape=(ny, nx), dtype=np.uint32)
        xmax = ymax = 3.0
        dx = 2.0 * xmax / float(nx)
        dy = 2.0 * ymax / float(ny)

        # make sure the radius for each bin is generated correctly
        posList = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)
        anglist = np.zeros(shape=(2), dtype=np.float32)

        # calculate the bin
        deltaX = posList[1][0] - posList[0][0]
        deltaY = posList[1][1] - posList[0][1]
        x = deltaX + xmax
        y = deltaY + ymax
        binX = numpy.floor(x / dx)
        binY = numpy.floor(y / dy)
        testArray[binY,binX] = 1

        # create object
        numNeighbors = 1
        EB = order.EntropicBonding(xmax, ymax, numNeighbors, testArray)

        # create the box
        box = trajectory.Box(Lx=5.0*xmax, Ly=5.0*ymax, is2D=True)

        # run the computation
        EB.compute(box, posList, anglist)

        # check to make sure that the point is in the correct bin

        bondList = EB.getBonds()

        npt.assert_equal(bondList[0][1][0], 1)

if __name__ == '__main__':
    unittest.main()
