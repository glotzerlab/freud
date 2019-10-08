import numpy as np
import numpy.testing as npt
import unittest


class TestManagedArray(object):
    def build_object(self):
        """Define how the compute class is built."""
        raise NotImplementedError(
            "Subclasses must define how the compute object is build.")

    def compute(self):
        """Define how the compute class's compute method is called."""
        raise NotImplementedError(
            "Subclasses must define how compute is called.")

    @property
    def computed_properties(self):
        """A list of strings indicating the computed properties to test."""
        raise NotImplementedError(
            "Subclasses must define the list of computed properties.")

    def test_saved_values(self):
        """Check that saved output don't get overwritten by later calls to
        compute or object deletion."""
        copied = []
        accessed = []
        self.build_object()

        copied = {}
        accessed = {}
        for prop in self.computed_properties:
            copied[prop] = []
            accessed[prop] = []

        num_tests = 25
        for i in range(num_tests):
            self.compute()

            for prop in self.computed_properties:
                copied[prop].append(np.copy(getattr(self.obj, prop)))
                accessed[prop].append(getattr(self.obj, prop))

        # Test that copying is not necessary.
        for prop in self.computed_properties:
            npt.assert_array_equal(copied[prop], accessed[prop])

        # Check that the memory outlives the originally owning object.
        del self.obj
        for prop in self.computed_properties:
            npt.assert_array_equal(copied[prop], accessed[prop])


if __name__ == '__main__':
    unittest.main()
