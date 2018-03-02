import numpy as np
import numpy.testing as npt
from freud import box, bond, parallel
import unittest

def genQ(axis=[0, 0, 1], angle=0.0):
    q = np.zeros(shape=(4), dtype=np.float32)
    q[0] = np.cos(0.5 * angle)
    q[1] = axis[0] * np.sin(0.5 * angle)
    q[2] = axis[1] * np.sin(0.5 * angle)
    q[3] = axis[2] * np.sin(0.5 * angle)
    q /= np.linalg.norm(q)
    return q

def quatMult(a, b):
    c = np.zeros(shape=(4), dtype=np.float32)
    c[0] = (a[0] * b[0]) - (a[1] * b[1]) - (a[2] * b[2]) - (a[3] * b[3])
    c[1] = (a[0] * b[1]) + (a[1] * b[0]) + (a[2] * b[3]) - (a[3] * b[2])
    c[2] = (a[0] * b[2]) - (a[1] * b[3]) + (a[2] * b[0]) + (a[3] * b[1])
    c[3] = (a[0] * b[3]) + (a[1] * b[2]) - (a[2] * b[1]) + (a[3] * b[0])
    return c

def quatRot(vec, q):
    qvert = [0.0, vec[0], vec[1], vec[2]]
    qs = np.copy(q)
    qs[1:] *= -1.0
    tmp = quatMult(q, qvert)
    rot = quatMult(tmp, qs)
    return np.asarray(rot[1:], dtype=np.float32)

class TestBond(unittest.TestCase):
    def test_correct_bond(self):
        # generate the bonding map
        nx = 70
        ny = 100
        nz = 50
        testArray = np.zeros(shape=(nz, ny, nx), dtype=np.uint32)
        xmax = 3.0
        ymax = 4.0
        zmax = 5.0
        dx = 2.0 * xmax / float(nx)
        dy = 2.0 * ymax / float(ny)
        dz = 2.0 * zmax / float(nz)

        # make sure the radius for each bin is generated correctly
        posList = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        qlist = np.zeros(shape=(2, 4), dtype=np.float32)
        qlist[:,0] = 1.0

        # calculate the bin
        deltaX = posList[1,0] - posList[0,0]
        deltaY = posList[1,1] - posList[0,1]
        deltaZ = posList[1,2] - posList[0,2]
        delta = np.array([deltaX, deltaY, deltaZ], dtype=np.float32)
        v_rot = quatRot(delta, qlist[0])
        x = v_rot[0] + xmax
        y = v_rot[1] + ymax
        z = v_rot[2] + zmax
        binX = int(np.floor(x / dx))
        binY = int(np.floor(y / dy))
        binZ = int(np.floor(z / dz))
        testArray[binZ,binY,binX] = 5
        deltaX = posList[0,0] - posList[1,0]
        deltaY = posList[0,1] - posList[1,1]
        deltaZ = posList[0,2] - posList[1,2]
        delta = np.array([deltaX, deltaY, deltaZ], dtype=np.float32)
        v_rot = quatRot(delta, qlist[1])
        x = v_rot[0] + xmax
        y = v_rot[1] + ymax
        z = v_rot[2] + zmax
        binX = int(np.floor(x / dx))
        binY = int(np.floor(y / dy))
        binZ = int(np.floor(z / dz))
        testArray[binZ,binY,binX] = 5

        # create object
        bondList = np.array([0, 5], dtype=np.uint32)
        EB = bond.BondingXYZ(xmax, ymax, zmax, testArray, bondList)

        # create the box
        rmax = np.sqrt(xmax**2 + ymax**2 + zmax**2)
        f_box = box.Box.cube(5.0*rmax)

        # run the computation
        EB.compute(f_box, posList, qlist, posList, qlist)

        # check to make sure that the point is in the correct bin

        bonds = EB.bonds

        npt.assert_equal(bonds[0,1], 1)
        npt.assert_equal(bonds[1,1], 0)

    def test_mapping(self):
        # generate the bonding map
        nx = 70
        ny = 100
        nz = 50
        testArray = np.zeros(shape=(nz, ny, nx), dtype=np.uint32)
        xmax = 3.0
        ymax = 4.0
        zmax = 5.0
        dx = 2.0 * xmax / float(nx)
        dy = 2.0 * ymax / float(ny)
        dz = 2.0 * zmax / float(nz)

        # create object
        bondList = np.array([0, 3, 4, 5], dtype=np.uint32)
        EB = bond.BondingXYZ(xmax, ymax, zmax, testArray, bondList)

        bond_map = EB.list_map

        npt.assert_equal(bond_map[3], 1)

    def test_rev_mapping(self):
        # generate the bonding map
        nx = 70
        ny = 100
        nz = 50
        testArray = np.zeros(shape=(nz, ny, nx), dtype=np.uint32)
        xmax = 3.0
        ymax = 4.0
        zmax = 5.0
        dx = 2.0 * xmax / float(nx)
        dy = 2.0 * ymax / float(ny)
        dz = 2.0 * zmax / float(nz)

        # create object
        bondList = np.array([0, 3, 4, 5], dtype=np.uint32)
        EB = bond.BondingXYZ(xmax, ymax, zmax, testArray, bondList)

        bond_map = EB.rev_list_map

        npt.assert_equal(bond_map[1], bondList[1])

if __name__ == '__main__':
    unittest.main()
