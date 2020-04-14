import freud
import unittest


class TestDiffractionPattern(unittest.TestCase):
    def test_compute(self):
        box, positions = freud.data.UnitCell.fcc().generate_system(4)

    def test_attribute_access(self):
        dp = freud.diffraction.DiffractionPattern()

        with self.assertRaises(AttributeError):
            dp.diffraction

        box, positions = freud.data.UnitCell.fcc().generate_system(4)
        dp.compute((box, positions))

        dp.diffraction

    def test_repr(self):
        dp = freud.diffraction.DiffractionPattern()
        self.assertEqual(str(dp), str(eval(repr(dp))))

        # Use non-default arguments for all parameters
        dp = freud.diffraction.DiffractionPattern(grid_size=500, zoom=1,
                                                  peak_width=2, bot=1e-6,
                                                  top=0.1)
        self.assertEqual(str(dp), str(eval(repr(dp))))


if __name__ == '__main__':
    unittest.main()
