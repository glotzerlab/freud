# Copyright (c) 2010-2024 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import numpy as np
import numpy.testing as npt


class ManagedArrayTestBase:
    def build_object(self):
        """Define how the compute class is built."""
        msg = "Subclasses must define how the compute object is built."
        raise NotImplementedError(
            msg
        )

    def compute(self):
        """Define how the compute class's compute method is called."""
        msg = "Subclasses must define how compute is called."
        raise NotImplementedError(msg)

    @property
    def computed_properties(self):
        """A list of strings indicating the computed properties to test."""
        msg = "Subclasses must define the list of computed properties."
        raise NotImplementedError(
            msg
        )

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
        for _i in range(num_tests):
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
