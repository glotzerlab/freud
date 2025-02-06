// Copyright (c) 2010-2025 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <complex>
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly

#include "VectorMath.h"
#include "export-ManagedArray.h"

namespace freud::util::detail {
void export_ManagedArray(nanobind::module_& module)
{ // python wrapper classes for ManagedArray
    export_ManagedArray<float>(module, "ManagedArray_float");
    export_ManagedArray<double>(module, "ManagedArray_double");
    export_ManagedArray<std::complex<float>>(module, "ManagedArray_complexfloat");
    export_ManagedArray<unsigned int>(module, "ManagedArray_unsignedint");
    export_ManagedArray<vec3<float>>(module, "ManagedArrayVec3_float");
    export_ManagedArray<std::complex<float>>(module, "ManagedArray_complexfloat");
    export_ManagedArray<std::complex<double>>(module, "ManagedArray_complexdouble");
    export_ManagedArray<char>(module, "ManagedArray_char");
};

}; // namespace freud::util::detail
