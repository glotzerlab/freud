// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly

#include "VectorMath.h"
#include "export-ManagedArray.h"

namespace freud::util::detail {
void export_ManagedArray(nanobind::module_& module)
{ // python wrapper classes for ManagedArray
    export_ManagedArray<float>(module, "ManagedArray_float");
    export_ManagedArray<double>(module, "ManagedArray_double");
    export_ManagedArray<unsigned int>(module, "ManagedArray_unsignedint");
    export_ManagedArray<vec3<float>>(module, "ManagedArrayVec3_float");
};

}; // namespace freud::util::detail