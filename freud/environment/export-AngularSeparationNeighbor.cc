// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly
#include <nanobind/stl/vector.h>     // NOLINT(misc-include-cleaner): used implicitly
#include <vector>

#include "AngularSeparation.h"

namespace nb = nanobind;

namespace freud { namespace environment {

namespace wrap {

}

namespace detail {

void export_AngularSeparationNeighbor(nb::module_& module)
{
    nb::class_<AngularSeparationNeighbor>(module, "AngularSeparationNeighbor")
        .def("getNList", &AngularSeparationNeighbor::getNList)
        .def("getAngles", &AngularSeparationNeighbor::getAngles)
        .def("compute", &AngularSeparationNeighbor::compute);
}

}; // namespace detail

}; }; // namespace freud::locality
