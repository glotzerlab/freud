#include <boost/python.hpp>

#include "trajectory.h"
#include "num_util.h"
#include "LinkCell.h"

using namespace boost::python;
namespace bnp=boost::python::numeric;

BOOST_PYTHON_MODULE(_freud)
    {
    // setup needed for numpy
    import_array();
    bnp::array::set_module_and_type("numpy", "ndarray");

    export_trajectory();
    export_LinkCell();
    }
