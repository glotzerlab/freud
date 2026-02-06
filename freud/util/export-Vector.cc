// Copyright (c) 2010-2026 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly

#include "VectorMath.h"
#include "export-Vector.h"

namespace freud::util::detail {

void export_Vector(nanobind::module_& module)
{ // python wrapper class for Vector
    export_Vector<float>(module, "Vector_float");
    export_Vector<double>(module, "Vector_double");
    export_Vector<unsigned int>(module, "Vector_unsignedint");
    export_Vector<vec3<float>>(module, "VectorVec3_float");
};

}; // namespace freud::util::detail
