#include <boost/python.hpp>

#include "trajectory.h"
#include "num_util.h"
#include "LinkCell.h"
#include "Cluster.h"
#include "GaussianDensity.h"
#include "RDF.h"
#include "ClusterProperties.h"
#include "HexOrderParameter.h"
#include "InterfaceMeasure.h"

using namespace boost::python;
namespace bnp=boost::python::numeric;
using namespace freud;

int my_import_array()
    {
    import_array();
    }

BOOST_PYTHON_MODULE(_freud)
    {
    // setup needed for numpy
    my_import_array();
    bnp::array::set_module_and_type("numpy", "ndarray");

    trajectory::export_trajectory();
    locality::export_LinkCell();
    density::export_RDF();
    density::export_GaussianDensity();
    cluster::export_Cluster();
    cluster::export_ClusterProperties();
    order::export_HexOrderParameter();
    interface::export_InterfaceMeasure();
    }
