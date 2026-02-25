// Copyright (c) 2010-2026 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>      // NOLINT(misc-include-cleaner): used implicitly
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly

#include "PeriodicBuffer.h"

namespace nb = nanobind;

namespace freud { namespace locality { namespace detail {

void export_PeriodicBuffer(nb::module_& module)
{
    nb::class_<PeriodicBuffer>(module, "PeriodicBuffer")
        .def(nb::init<>())
        .def("compute", &PeriodicBuffer::compute)
        .def("getBox", &PeriodicBuffer::getBox)
        .def("getBufferBox", &PeriodicBuffer::getBufferBox)
        .def("getBufferPoints", &PeriodicBuffer::getBufferPoints)
        .def("getBufferIds", &PeriodicBuffer::getBufferIds);
};

}; }; }; // namespace freud::locality::detail
