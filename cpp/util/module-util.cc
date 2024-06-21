// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>

#include "export_ManagedArray.h"

using namespace freud::util::detail;

NB_MODULE(_util, m)
{
    export_ManagedArray<float>(m, "ManagedArray_float");
    export_ManagedArray<double>(m, "ManagedArray_double");
    export_ManagedArray<unsigned int>(m, "ManagedArray_unsignedint");
    export_ManagedArrayVec3<float>(m, "ManagedArrayVec3_float");
}
