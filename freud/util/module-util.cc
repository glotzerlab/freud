// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>
#include <nanobind/nb_defs.h>

#include "export-ManagedArray.h"
#include "export-Vector.h"

using namespace freud::util::detail;

NB_MODULE(_util, module) // NOLINT(misc-use-anonymous-namespace): We have no control over nanobind module definitions.
{
    // python wrapper classes for ManagedArray
    export_ManagedArray<float>(module, "ManagedArray_float");
    export_ManagedArray<double>(module, "ManagedArray_double");
    export_ManagedArray<unsigned int>(module, "ManagedArray_unsignedint");
    export_ManagedArray<vec3<float>>(module, "ManagedArrayVec3_float");

    // python wrapper class for Vector
    export_Vector<float>(module, "Vector_float");
    export_Vector<double>(module, "Vector_double");
    export_Vector<unsigned int>(module, "Vector_unsignedint");
    export_Vector<vec3<float>>(module, "VectorVec3_float");
}
