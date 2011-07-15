#include <boost/python.hpp>

#include "trajectory.h"
#include "num_util.h"
#include "LinkCell.h"
#include "Cluster.h"
#include "RDF.h"
#include "ClusterProperties.h"

using namespace boost::python;
namespace bnp=boost::python::numeric;
using namespace freud;

BOOST_PYTHON_MODULE(_freud)
    {
    // setup needed for numpy
    import_array();
    bnp::array::set_module_and_type("numpy", "ndarray");

    trajectory::export_trajectory();
    locality::export_LinkCell();
    density::export_RDF();
    cluster::export_Cluster();
    cluster::export_ClusterProperties();
    }
