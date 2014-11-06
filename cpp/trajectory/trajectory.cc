#include <iostream>

#include "num_util.h"
#include "trajectory.h"
#include "DCDLoader.h"
#include "Index1D.h"

using namespace std;
using namespace boost::python;
namespace bnp=boost::python::numeric;

/*! \file trajectory.h
    \brief Helper routines for trajectory
*/

namespace freud { namespace trajectory {

void export_trajectory()
    {
    // define functions
    class_<Box>("Box", init<float, optional<bool> >())
        .def(init<float, float, float, optional<bool> >())
        .def(init<float, float, float, float, float, float, optional<bool> >())
        .def("set2D", &Box::set2D)
        .def("is2D", &Box::is2D)
        .def("getLx", &Box::getLx)
        .def("getLy", &Box::getLy)
        .def("getLz", &Box::getLz)
        .def("getL", &Box::getL)
        .def("getLinv", &Box::getLinv)
        .def("getTiltFactorXY", &Box::getTiltFactorXY)
        .def("getTiltFactorXZ", &Box::getTiltFactorXZ)
        .def("getTiltFactorYZ", &Box::getTiltFactorYZ)
        .def("getVolume", &Box::getVolume)
        .def("getCoordinates", &Box::getCoordinatesPy)
        .def("wrap", &Box::wrapPy)
        .enable_pickling()
        ;
    export_dcdloader();
    }

}; }; // end namespace freud::trajectory
