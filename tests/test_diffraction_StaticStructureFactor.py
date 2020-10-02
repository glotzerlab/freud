import freud
import matplotlib
import unittest
# import numpy as np
# import numpy.testing as npt
matplotlib.use('agg')


class TestStaticStructureFactor(unittest.TestCase):
    def test_compute(self):
        sf = freud.diffraction.StaticStructureFactor(1000, 100)
        box, positions = freud.data.UnitCell.fcc().generate_system(4)
        sf.compute((box, positions))

# TODO: All the below tests were copied from DiffractionPattern and need to be
# updated for this class

#    def test_attribute_access(self):
#        grid_size = 234
#        output_size = 123
#        sf = freud.diffraction.StaticStructureFactor(1000, 100)
#        self.assertEqual(sf.grid_size, grid_size)
#        self.assertEqual(sf.output_size, grid_size)
#        sf = freud.diffraction.StaticStructureFactor(
#            grid_size=grid_size, output_size=output_size)
#        self.assertEqual(sf.grid_size, grid_size)
#        self.assertEqual(sf.output_size, output_size)
#
#        box, positions = freud.data.UnitCell.fcc().generate_system(4)
#
#        with self.assertRaises(AttributeError):
#            sf.diffraction
#        with self.assertRaises(AttributeError):
#            sf.k_values
#        with self.assertRaises(AttributeError):
#            sf.k_vectors
#        with self.assertRaises(AttributeError):
#            sf.plot()
#
#        sf.compute((box, positions), zoom=1, peak_width=4)
#        diff = sf.diffraction
#        vals = sf.k_values
#        vecs = sf.k_vectors
#        sf.plot()
#        sf._repr_png_()
#
#        # Make sure old data is not invalidated by new call to compute()
#        box2, positions2 = freud.data.UnitCell.bcc().generate_system(3)
#        sf.compute((box2, positions2), zoom=1, peak_width=4)
#        self.assertFalse(np.array_equal(sf.diffraction, diff))
#        self.assertFalse(np.array_equal(sf.k_values, vals))
#        self.assertFalse(np.array_equal(sf.k_vectors, vecs))
#
#    def test_attribute_shapes(self):
#        grid_size = 234
#        output_size = 123
#        sf = freud.diffraction.StaticStructureFactor(
#            grid_size=grid_size, output_size=output_size)
#        box, positions = freud.data.UnitCell.fcc().generate_system(4)
#        sf.compute((box, positions))
#
#        self.assertEqual(sf.diffraction.shape, (output_size, output_size))
#        self.assertEqual(sf.k_values.shape, (output_size,))
#        self.assertEqual(sf.k_vectors.shape, (output_size, output_size, 3))
#        self.assertEqual(sf.to_image().shape, (output_size, output_size, 4))
#
#    def test_repr(self):
#        sf = freud.diffraction.StaticStructureFactor()
#        self.assertEqual(str(sf), str(eval(repr(sf))))
#
#        # Use non-default arguments for all parameters
#        sf = freud.diffraction.StaticStructureFactor(
#            grid_size=123, output_size=234)
#        self.assertEqual(str(sf), str(eval(repr(sf))))
#
#    def test_k_values_and_k_vectors(self):
#        sf = freud.diffraction.StaticStructureFactor()
#
#        for size in [2, 5, 10]:
#            for npoints in [10, 20, 75]:
#                box, positions = freud.data.make_random_system(npoints, size)
#                sf.compute((box, positions))
#
#                output_size = sf.output_size
#                npt.assert_allclose(sf.k_values[output_size//2], 0)
#                center_index = (output_size//2, output_size//2)
#                npt.assert_allclose(sf.k_vectors[center_index], [0, 0, 0])


if __name__ == '__main__':
    unittest.main()
