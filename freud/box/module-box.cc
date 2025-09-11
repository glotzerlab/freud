// Copyright (c) 2010-2025 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>
#include <nanobind/nb_defs.h>
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly
#include <nanobind/stl/vector.h>     // NOLINT(misc-include-cleaner): used implicitly

#include <utility>

#include "Box.h"
#include "export-Box.h"

using namespace freud::box;

NB_MODULE(_box, module) // NOLINT(misc-use-anonymous-namespace, modernize-avoid-c-arrays): caused by nanobind
{
    nanobind::class_<Box>(std::move(module), "Box")
        // constructors
        .def(nanobind::init<float, float, float, float, float, float, bool>())
        // getters and setters
        .def("getLx", &Box::getLx)
        .def("getLy", &Box::getLy)
        .def("getLz", &Box::getLz)
        .def("setL", &Box::setL)
        .def("getLinv", &Box::getLinv)
        .def("getTiltFactorXY", &Box::getTiltFactorXY)
        .def("getTiltFactorXZ", &Box::getTiltFactorXZ)
        .def("getTiltFactorYZ", &Box::getTiltFactorYZ)
        .def("setTiltFactorXY", &Box::setTiltFactorXY)
        .def("setTiltFactorXZ", &Box::setTiltFactorXZ)
        .def("setTiltFactorYZ", &Box::setTiltFactorYZ)
        .def("getPeriodicX", &Box::getPeriodicX)
        .def("getPeriodicY", &Box::getPeriodicY)
        .def("getPeriodicZ", &Box::getPeriodicZ)
        .def("setPeriodic", &Box::setPeriodic)
        .def("setPeriodicX", &Box::setPeriodicX)
        .def("setPeriodicY", &Box::setPeriodicY)
        .def("setPeriodicZ", &Box::setPeriodicZ)
        .def("is2D", &Box::is2D)
        .def("set2D", &Box::set2D)
        .def("getVolume", &Box::getVolume)
        .def("center", &wrap::center)
        .def("centerOfMass", &wrap::centerOfMass)
        // other stuff
        .def("makeAbsolute", &wrap::makeAbsolute)
        .def("makeFractional", &wrap::makeFractional)
        .def("wrap", &wrap::wrap)
        .def("unwrap", &wrap::unwrap)
        .def("getImages", &wrap::getImages)
        .def("computeDistances", &wrap::computeDistances)
        .def("computeAllDistances", &wrap::computeAllDistances)
        .def("contains", &wrap::contains);
}
