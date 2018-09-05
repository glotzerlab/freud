import freud
import unittest
from freud.errors import FreudDeprecationWarning
import warnings


class TestIndex(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=FreudDeprecationWarning)

    def test_index_2d(self):
        """Test that indexing works as expected."""
        N = 10
        idx = freud.index.Index2D(N)
        for i in range(N):
            for j in range(N):
                self.assertEqual(idx(i, j), N*j+i)
        self.assertEqual(N*N, idx.num_elements)
        self.assertEqual(N*N, idx.getNumElements())

    def test_index_3d(self):
        """Test that indexing works as expected."""
        N = 10
        idx = freud.index.Index3D(N)
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    self.assertEqual(idx(i, j, k), N*N*k+N*j+i)

        self.assertEqual(N*N*N, idx.num_elements)
        self.assertEqual(N*N*N, idx.getNumElements())


if __name__ == '__main__':
    unittest.main()
