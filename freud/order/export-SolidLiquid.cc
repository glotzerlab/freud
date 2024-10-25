// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly
#include <utility>

// #include "ManagedArray.h"
#include "SolidLiquid.h"
// #include "VectorMath.h"


namespace freud { namespace order {

namespace wrap {}; // namespace wrap

namespace detail {

void export_SolidLiquid(nanobind::module_& m)
{
    nanobind::class_<SolidLiquid>(m, "SolidLiquid")
        .def(nanobind::init<unsigned int, float, unsigned int, bool>())
        .def("compute", &SolidLiquid::compute, nanobind::arg("nlist").none(), nanobind::arg("points"), nanobind::arg("qargs"))
        ;
}

} // namespace detail

}; }; // namespace freud::order
