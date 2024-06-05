// Copyright (c) 2010-2023 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "Box.h"

using namespace freud::box;

NB_MODULE(_box, m)
{
    nanobind::class_<Box>(m, "Box")
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
        .def("center", &Box::center)
        .def("centerOfMass", &Box::centerOfMassPython)
        // other stuff
        .def("makeAbsolute", &Box::makeAbsolutePython)
        .def("makeFractional", &Box::makeFractionalPython)
        .def("wrap", &Box::wrapPython)
        .def("unwrap", &Box::unwrap)
        .def("getImages", &Box::getImages)
        .def("computeDistances", &Box::computeDistances)
        .def("computeAllDistances", &Box::computeAllDistances)
        .def("contains", &Box::contains);
}
