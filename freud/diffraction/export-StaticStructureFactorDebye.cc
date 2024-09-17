// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>

#include "StaticStructureFactorDebye.h"

namespace freud { namespace diffraction {

    namespace detail {

        void export_StaticStructureFactorDebye(nanobind::module_& m)
        {
            nanobind::class_<StaticStructureFactorDebye, StaticStructureFactor>(m, "StaticStructureFactorDebye")
                .def(nanobind::init<unsigned int, float, float>());
        }

    } // namespace detail
}}