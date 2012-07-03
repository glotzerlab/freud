#include <iostream>

#include "num_util.h"
#include "trajectory.h"
#include "DCDLoader.h"
#include "Index1D.h"

using namespace std;
using namespace boost::python;
namespace bnp=boost::python::numeric;

namespace freud { namespace trajectory {

void export_trajectory()
    {
    // define functions
    class_<Box>("Box", init<float, optional<bool> >())
        .def(init<float, float, float, optional<bool> >())
        .def("set2D", &Box::set2D)
        .def("is2D", &Box::is2D)
        .def("getLx", &Box::getLx)
        .def("getLy", &Box::getLy)
        .def("getLz", &Box::getLz)
        .def("getVolume", &Box::getVolume)
        .def("wrap", &Box::wrapPy)
        /*.def("unwrap", &Box::unwrapPy)
        .def("makeunit", &Box::makeunitPy)*/
        ;
    export_dcdloader();
    }

}; }; // end namespace freud::trajectory