import numpy as np
import numpy.testing as npt
import freud
import unittest
import util

class TestSymmetryCollection(unittest.TestCase):
    def test_maxL(self):
        N = 1000
        box = freud.box.Box.cube(10)
        np.random.seed(0)
        positions = np.random.uniform(-box.getLx()/2, box.getLx()/2, size=(N, 3)).astype(np.float32)

        #test default maxL
        sc = freud.symmetry.SymmetryCollection()
        npt.assert_equal(sc.getMaxL(), 30)

        #set maxL to a value
        sc = freud.symmetry.SymmetryCollection(1)
        npt.assert_equal(sc.getMaxL(), 1)

    def test_rotate(self):
        #(box, positions) = util.make_fcc(4, 4, 4)

       # comp = freud.order.LocalQl(box, 1.5, 6)

        #comp.compute(positions)
        #assert np.allclose(comp.Ql, comp.Ql[0])

        #comp.computeAve(positions)
        #assert np.allclose(comp.ave_Ql, comp.ave_Ql[0])

        #comp.computeNorm(positions)
        #assert np.allclose(comp.norm_Ql, comp.norm_Ql[0])

        #comp.computeAveNorm(positions)
        #assert np.allclose(comp.ave_norm_Ql, comp.ave_norm_Ql[0])


        #test compute function
        #make a fcc cube
        (box, positions) = util.make_fcc(4, 4, 4)
        nn = freud.locality.NearestNeighbors(2.0, 12)
        nlist = nn.compute(box, positions, positions).nlist

        sc = freud.symmetry.SymmetryCollection(30)
        sc.compute(box, positions, nlist)

        sc_rotate = freud.symmetry.SymmetryCollection(30)
        sc_rotate.compute(box, positions, nlist)

        q = np.array([1,0,0,0], dtype = np.float32)
        sc_rotate.rotate(q)

        #rotate by a unit quaternion, test rotate and measure functions
        for t in range(-1, 7):
            print(t, sc.measure(t), sc_rotate.measure(t))
            #assert np.isclose(sc.measure(t), sc_rotate.measure(t), rtol=5e-4, atol=5e-5)

        #test getMlm and getMlm_rotated functions
        assert np.allclose(sc_rotate.getMlm(), sc_rotate.getMlm_rotated())

    def test_measure(self):
        pass
        #(box, positions) = util.make_fcc(4, 4, 4)

        #omp = freud.order.LocalQl(box, 1.5, 6)

        #comp.compute(positions)
        #assert np.allclose(comp.Ql, comp.Ql[0])

        #comp.computeAve(positions)
        #assert np.allclose(comp.ave_Ql, comp.ave_Ql[0])

        #comp.computeNorm(positions)
        #assert np.allclose(comp.norm_Ql, comp.norm_Ql[0])

        #comp.computeAveNorm(positions)
        #assert np.allclose(comp.ave_norm_Ql, comp.ave_norm_Ql[0])


        #test rotate function
        #make a fcc cube
        (box, positions) = util.make_fcc(4, 4, 4)
        nn = freud.locality.NearestNeighbors(2.0, 12)
        nlist = nn.compute(box, positions, positions).nlist
        sc = freud.symmetry.SymmetryCollection(30)
        sc_rotate = freud.symmetry.SymmetryCollection(30)
        sc.compute(box, positions, nlist)
        sc_rotate.compute(box, positions, nlist)

        #unit quaternion
        sc_rotate.rotate([1,0,0,0])
        #print()
        #for t in range(-1, 7):
            #print(t, sc.measure(t), sc_rotate.measure(t))


        #npt.assert_equal(sc.measure(-1), )

    def test_symmetries(self):
        # Make an FCC lattice
        (box, positions) = util.make_fcc(4, 4, 4)
        nn = freud.locality.NearestNeighbors(2.0, 12)
        nlist = nn.compute(box, positions, positions).nlist
        sc = freud.symmetry.SymmetryCollection(30)
        sc.compute(box, positions, nlist)
        #for symm in sc.symmetries:
         #   print(symm)

    def test_laueGroup(self):
        # Make an FCC lattice
        (box, positions) = util.make_fcc(4, 4, 4)
        nn = freud.locality.NearestNeighbors(2.0, 12)
        nlist = nn.compute(box, positions, positions).nlist
        sc = freud.symmetry.SymmetryCollection(30)
        sc.compute(box, positions, nlist)

        # Ensure the Laue Group of an FCC crystal is m-3m
        assert sc.getLaueGroup() == 'm-3m'


class TestLocalQlNear(unittest.TestCase):
    def test_shape(self):
        pass
        N = 1000

        box = freud.box.Box.cube(10)
        np.random.seed(0)
        positions = np.random.uniform(-box.getLx()/2, box.getLx()/2, size=(N, 3)).astype(np.float32)

        comp = freud.order.LocalQlNear(box, 1.5, 6, 12)
        comp.compute(positions)

        npt.assert_equal(comp.Ql.shape[0], N)

    def test_identical_environments(self):
        pass
        (box, positions) = util.make_fcc(4, 4, 4)

        comp = freud.order.LocalQlNear(box, 1.5, 6, 12)

        comp.compute(positions)
        assert np.allclose(comp.Ql, comp.Ql[0])

        comp.computeAve(positions)
        assert np.allclose(comp.ave_Ql, comp.ave_Ql[0])

        comp.computeNorm(positions)
        assert np.allclose(comp.norm_Ql, comp.norm_Ql[0])

        comp.computeAveNorm(positions)
        assert np.allclose(comp.ave_norm_Ql, comp.ave_norm_Ql[0])

if __name__ == '__main__':
    unittest.main()
