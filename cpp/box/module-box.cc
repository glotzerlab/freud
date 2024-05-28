
#include <pybind11/pybind11.h>

#include "Box.h"

PYBIND11_MODULE(_box, m)
{
    pybind11::class_<Box, std::shared_ptr<Box>>(m, "Box")
        // constructors
        .def(pybind11::init<float, float, float, float, float, float, bool>())
        // getters and setters
        .def("getLx", &Box::getLx)
        .def("getLy", &Box::getLy)
        .def("getLz", &Box::getLz)
        .def("setL", &Box::setL)
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
        .def("get2D", &Box::get2D)
        .def("set2D", &Box::set2D)
        .def("getVolume", &Box::getVolume)
        .def("center", &Box::center)
        .def("centerOfMass", &Box::centerOfMass)
        // other stuff
        .def("makeAbsolute", &Box::makeAbsolute)
        .def("makeFractional", &Box::makeFractional)
        .def("wrap", &Box::wrap)
        .def("unwrap", &Box::unwrap)
        .def("computeDistances", &Box::computeDistances)
        .def("computeAllDistances", &Box::computeAllDistances)
        .def("contains", &Box::contains);
}
