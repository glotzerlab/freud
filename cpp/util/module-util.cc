// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>

#include "export_ManagedArray.h"
#include "export_Vector.h"

using namespace freud::util::detail;

NB_MODULE(_util, m)
{
    // python wrapper classes for ManagedArray
    export_ManagedArray<float>(m, "ManagedArray_float");
    export_ManagedArray<double>(m, "ManagedArray_double");
    export_ManagedArray<unsigned int>(m, "ManagedArray_unsignedint");
    export_ManagedArray<vec3<float>>(m, "ManagedArrayVec3_float");

    // python wrapper class for Vector
    export_Vector<float>(m, "Vector_float");
    export_Vector<double>(m, "Vector_double");
    export_Vector<unsigned int>(m, "Vector_unsignedint");
    export_VectorVec3<float>(m, "VectorVec3_float");
}
