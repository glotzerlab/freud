from freud import locality, trajectory
import numpy as np
import numpy.testing as npt
import unittest

class TestLinkCell(unittest.TestCase):
    def test_unique_neighbors(self):
        L = 10 #Box Dimensions
        rcut = 3 #Cutoff radius

        box = trajectory.Box(L)#Initialize Box
        cl = locality.LinkCell(box,rcut)#Initialize cell list
        cl.computeCellList(np.zeros((1,3), dtype=np.float32))#Compute cell list

        # 27 is the total number of cells
        for i in range(27):
            neighbors = cl.getCellNeighbors(i)
            self.assertEqual(len(np.unique(neighbors)), 27,
                             msg="Cell %d does not have 27 unique adjacent cell indices, it has %d" % (i, len(np.unique(neighbors))))

    def test_bug2100(self):
        L = 10; #Box Dimensions
        rcut = 3; #Cutoff radius

        #Initialize test points across periodic BC
        testpoints = np.array([[-4.95,0,0],[4.95,0,0]], dtype=np.float32);
        box = trajectory.Box(L);#Initialize Box
        cl = locality.LinkCell(box,rcut);#Initialize cell list
        cl.computeCellList(testpoints);#Compute cell list

        #Get cell index
        cell_index0 = cl.getCell(testpoints[0])
        cell_index1 = cl.getCell(testpoints[1])

        #Get cell neighbors
        neighbors0 = cl.getCellNeighbors(cell_index0);
        neighbors1 = cl.getCellNeighbors(cell_index1);

        #Check if particle 0 is in a cell neighboring particle 1
        test0 = np.where(neighbors1 == cell_index0)[0]; #where returns [[index]] if found, otherwise [[]]
        test1 = np.where(neighbors0 == cell_index1)[0];
        self.assertEqual(len(test0), len(test1))


if __name__ == '__main__':
    unittest.main()



