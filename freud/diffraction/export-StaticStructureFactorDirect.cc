// Copyright (c) 2010-2025 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>        // NOLINT(misc-include-cleaner) used implicitly
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner) used implicitly

#include "StaticStructureFactor.h"
#include "StaticStructureFactorDirect.h"

namespace freud::diffraction::detail {

void export_StaticStructureFactorDirect(nanobind::module_& m)
{
    nanobind::class_<StaticStructureFactorDirect, StaticStructureFactor>(m, "StaticStructureFactorDirect")
        .def(nanobind::init<unsigned int, float, float, unsigned int>())
        .def("getNumSampledKPoints", &StaticStructureFactorDirect::getNumSampledKPoints)
        .def("getKPoints", &StaticStructureFactorDirect::getKPoints);
}

} // namespace freud::diffraction::detail
