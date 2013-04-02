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

/* numpy is terrible (see /opt/local/Library/Frameworks/Python.framework/Versions/2.7/
lib/python2.7/site-packages/numpy/core/generate_numpy_array.py)
The following #defines help get around this
*/

#if PY_VERSION_HEX >= 0x03000000
#define MY_PY_VER_3x
#else
#define MY_PY_VER_2x
#endif

#ifdef MY_PY_VER_3x
int my_import_array()
    {
    import_array();
    }
#endif
#ifdef MY_PY_VER_2x
void my_import_array()
    {
    import_array();
    }
#endif

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
