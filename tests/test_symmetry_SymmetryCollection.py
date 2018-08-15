import numpy as np
import numpy.testing as npt
import freud
import unittest
import util


class TestSymmetryCollection(unittest.TestCase):
    def test_maxL(self):
        # Test default maxL
        sc = freud.symmetry.SymmetryCollection()
        npt.assert_equal(sc.l_max, 30)

        # Set maxL to a value and make sure it is set
        sc = freud.symmetry.SymmetryCollection(1)
        npt.assert_equal(sc.l_max, 1)

    def test_rotate(self):
        (box, positions) = util.make_fcc(4, 4, 4)
        nn = freud.locality.NearestNeighbors(2.0, 12)
        nlist = nn.compute(box, positions, positions).nlist

        sc = freud.symmetry.SymmetryCollection(30)
        sc.compute(box, positions, nlist)

        sc_rotate = freud.symmetry.SymmetryCollection(30)
        sc_rotate.compute(box, positions, nlist)

        q = np.array([1, 0, 0, 0], dtype=np.float32)
        sc_rotate.rotate(q)

        # Rotate by a unit quaternion, test rotate and measure functions
        for t in range(-1, 7):
            print(t, sc.measure(t), sc_rotate.measure(t))

        # Test getMlm and getMlm_rotated functions
        assert np.allclose(sc_rotate.getMlm(), sc_rotate.getMlm_rotated())

    def test_symmetries(self):
        # Make an FCC lattice
        (box, positions) = util.make_fcc(4, 4, 4)
        nn = freud.locality.NearestNeighbors(2.0, 12)
        nlist = nn.compute(box, positions, positions).nlist
        sc = freud.symmetry.SymmetryCollection(30)
        sc.compute(box, positions, nlist)

    def test_laueGroup(self):
        # Make an FCC lattice
        (box, positions) = util.make_fcc(4, 4, 4)
        nn = freud.locality.NearestNeighbors(2.0, 12)
        nlist = nn.compute(box, positions, positions).nlist
        sc = freud.symmetry.SymmetryCollection(30)
        sc.compute(box, positions, nlist)

        # Ensure the Laue Group of an FCC crystal is m-3m
        assert sc.getLaueGroup() == 'm-3m'


if __name__ == '__main__':
    unittest.main()
