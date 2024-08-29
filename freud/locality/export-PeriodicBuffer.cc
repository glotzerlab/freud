// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <nanobind/stl/array.h>
#include <nanobind/stl/shared_ptr.h>

#include "PeriodicBuffer.h"

namespace nb = nanobind;

namespace freud { namespace locality { namespace detail {

void export_PeriodicBuffer(nb::module_& m)
{
    nb::class_<PeriodicBuffer>(m, "PeriodicBuffer")
        .def(nb::init<>())
        .def("compute", &PeriodicBuffer::compute)
        .def("getBox", &PeriodicBuffer::getBox)
        .def("getBufferBox", &PeriodicBuffer::getBufferBox)
        .def("getBufferPoints", &PeriodicBuffer::getBufferPoints)
        .def("getBufferIds", &PeriodicBuffer::getBufferIds);
};

}; }; }; // namespace freud::locality::detail
