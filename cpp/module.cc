#include <boost/python.hpp>

#include "trajectory.h"
#include "num_util.h"
#include "LinkCell.h"
#include "Cluster.h"
#include "GaussianDensity.h"
#include "RDF.h"
#include "ClusterProperties.h"

using namespace boost::python;
namespace bnp=boost::python::numeric;

BOOST_PYTHON_MODULE(_freud)
    {
    // setup needed for numpy
    import_array();
    bnp::array::set_module_and_type("numpy", "ndarray");

    export_trajectory();
    export_GaussianDensity();
		export_LinkCell();
    export_RDF();
    export_Cluster();
    export_ClusterProperties();
    }
