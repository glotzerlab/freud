import freud
import unittest


class TestIndex(unittest.TestCase):
    def test_index_2d(self):
        """Test that indexing works as expected."""
        N = 10
        idx = freud.index.Index2D(N)
        for i in range(N):
            for j in range(N):
                self.assertEqual(idx(i, j), N*j+i)
        self.assertEqual(N*N, idx.num_elements)

    def test_index_3d(self):
        """Test that indexing works as expected."""
        N = 10
        idx = freud.index.Index3D(N)
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    self.assertEqual(idx(i, j, k), N*N*k+N*j+i)

        self.assertEqual(N*N*N, idx.num_elements)

    def test_repr(self):
        two_d = freud.index.Index2D(4, 5)
        self.assertEqual(str(two_d), str(eval(repr(two_d))))
        three_d = freud.index.Index3D(4, 5, 6)
        self.assertEqual(str(three_d), str(eval(repr(three_d))))


if __name__ == '__main__':
    unittest.main()
