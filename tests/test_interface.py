import numpy as np
import freud
import unittest
import util


class TestInterface(unittest.TestCase):
    def test_constructor(self):
        """Ensure no arguments are accepted to the constructor."""
        freud.interface.Interface()
        with self.assertRaises(TypeError):
            freud.interface.Interface(0)

    def test_take_one(self):
        """Test that there is exactly 1 or 12 particles at the interface when
        one particle is removed from an FCC structure"""
        np.random.seed(0)
        (box, positions) = util.make_fcc(4, 4, 4, noise=1e-2)
        positions.flags['WRITEABLE'] = False

        index = np.random.randint(0, len(positions))

        point = positions[index].reshape((1, 3))
        others = np.concatenate([positions[:index], positions[index + 1:]])

        inter = freud.interface.Interface()

        # Test attribute access
        with self.assertRaises(AttributeError):
            inter.point_count
        with self.assertRaises(AttributeError):
            inter.point_ids
        with self.assertRaises(AttributeError):
            inter.query_point_count
        with self.assertRaises(AttributeError):
            inter.query_point_ids

        test_one = inter.compute(
            (box, point), others, neighbors=dict(r_max=1.5))

        # Test attribute access
        inter.point_count
        inter.point_ids
        inter.query_point_count
        inter.query_point_ids

        self.assertEqual(test_one.point_count, 1)
        self.assertEqual(len(test_one.point_ids), 1)

        test_twelve = inter.compute(
            (box, others), point, neighbors=dict(r_max=1.5))
        self.assertEqual(test_twelve.point_count, 12)
        self.assertEqual(len(test_twelve.point_ids), 12)

    def test_filter_r(self):
        """Test that nlists are filtered to the correct r_max."""
        np.random.seed(0)
        r_max = 3.0
        (box, positions) = util.make_fcc(4, 4, 4, noise=1e-2)

        index = np.random.randint(0, len(positions))

        point = positions[index].reshape((1, 3))
        others = np.concatenate([positions[:index], positions[index + 1:]])

        # Creates a NeighborList with r_max larger than the interface size
        aq = freud.locality.AABBQuery(box, others)
        nlist = aq.query(point, dict(r_max=r_max)).toNeighborList()

        # Filter NeighborList
        nlist.filter_r(1.5)

        inter = freud.interface.Interface()

        test_twelve = inter.compute(
            (box, others), point, nlist)
        self.assertEqual(test_twelve.point_count, 12)
        self.assertEqual(len(test_twelve.point_ids), 12)

    def test_repr(self):
        inter = freud.interface.Interface()
        self.assertEqual(str(inter), str(eval(repr(inter))))


if __name__ == '__main__':
    unittest.main()
