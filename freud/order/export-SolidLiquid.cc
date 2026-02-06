// Copyright (c) 2010-2026 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly

#include "SolidLiquid.h"

namespace freud::order::detail {

void export_SolidLiquid(nanobind::module_& m)
{
    nanobind::class_<SolidLiquid>(m, "SolidLiquid")
        .def(nanobind::init<unsigned int, float, unsigned int, bool>())
        .def("compute", &SolidLiquid::compute, nanobind::arg("nlist").none(), nanobind::arg("points"),
             nanobind::arg("qargs"))
        .def("getL", &SolidLiquid::getL)
        .def("getQThreshold", &SolidLiquid::getQThreshold)
        .def("getSolidThreshold", &SolidLiquid::getSolidThreshold)
        .def("getNormalizeQ", &SolidLiquid::getNormalizeQ)
        .def("getClusterIdx", &SolidLiquid::getClusterIdx)
        .def("getQlij", &SolidLiquid::getQlij)
        .def("getQlm", &SolidLiquid::getQlm)
        .def("getClusterSizes", &SolidLiquid::getClusterSizes)
        .def("getLargestClusterSize", &SolidLiquid::getLargestClusterSize)
        .def("getNList", &SolidLiquid::getNList)
        .def("getNumberOfConnections", &SolidLiquid::getNumberOfConnections);
}

}; // namespace freud::order::detail
