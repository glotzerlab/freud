import numpy
import numpy as np
# import numpy.testing as npt
from freud import trajectory, order, parallel
# import unittest

def test_correct_bond():
    # generate the bonding map
    nr = 10
    nt = 100
    testArray = np.zeros(shape=(nr, nt, nt), dtype=np.uint32)
    rmax = 4.0
    dr = rmax / float(nr)
    dt = 2.0 * np.pi / float(nt)
    print("python: dr = {}".format(dr))
    print("python: dt = {}".format(dt))

    # make sure the radius for each bin is generated correctly
    posList = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)
    anglist = np.zeros(shape=(2), dtype=np.float32)

    # calculate the bin
    deltaX = posList[1,0] - posList[0,0]
    deltaY = posList[1,1] - posList[0,1]
    delta = np.array([deltaX, deltaY], dtype=np.float32)
    print("python: delta = {}".format(delta))
    r = np.sqrt(np.dot(delta, delta))
    theta1 = anglist[0] - np.arctan2(deltaY, deltaX)
    theta2 = anglist[1] - np.arctan2(-deltaY, -deltaX)
    theta1 = theta1 if (theta1 > 0) else theta1 + 2.0*np.pi
    theta1 = theta1 if (theta1 < 2.0*np.pi) else theta1 - 2.0*np.pi
    theta2 = theta2 if (theta2 > 0) else theta2 + 2.0*np.pi
    theta2 = theta2 if (theta2 < 2.0*np.pi) else theta2 - 2.0*np.pi
    binR = numpy.floor(r / dr)
    binT1 = numpy.floor(theta1 / dt)
    binT2 = numpy.floor(theta2 / dt)
    print("python: binR = {}, binT1 = {}, binT2 = {}".format(binR, binT1, binT2))
    testArray[binR,binT1,binT2] = 1
    for i in range(testArray.shape[0]):
        for j in range(testArray.shape[1]):
            for k in range(testArray.shape[2]):
                if testArray[i,j,k] == 1:
                    print("python: val of 1 in {} {} {}".format(i, j, k))

    # create object
    numNeighbors = 1
    EB = order.EntropicBondingRT(rmax, numNeighbors, testArray)

    # create the box
    box = trajectory.Box(Lx=5.0*rmax, Ly=5.0*rmax, is2D=True)

    # run the computation
    EB.compute(box, posList, anglist)

    # check to make sure that the point is in the correct bin

    bondList = EB.getBonds()

    print(bondList)

    # npt.assert_equal(bondList[0][1][0], 1)

if __name__ == '__main__':
    test_correct_bond()
