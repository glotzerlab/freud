#include <tbb/tbb.h>
#include <ostream>

// work around nasty issue where python #defines isalpha, toupper, etc....
#undef __APPLE__
#include <Python.h>
#define __APPLE__

#include <boost/python.hpp>

#include "trajectory.h"
#include "num_util.h"
#include "LinkCell.h"
#include "Cluster.h"
#include "GaussianDensity.h"
#include "VoronoiBuffer.h"
#include "LocalDensity.h"
#include "kspace.h"
#include "RDF.h"
#include "ClusterProperties.h"
#include "HexOrderParameter.h"
#include "InterfaceMeasure.h"
#include "pairing2D.h"
#include "PMFXYZ.h"
#include "PMFXY2D.h"
#include "PMFTXYT2D.h"
#include "PMFTXYTP2D.h"
#include "PMFTXYTM2D.h"
#include "PMFTRPM.h"
#include "PMFTR12.h"
#include "Bootstrap.h"
#include "shapesplit.h"
#include "colormap.h"
#include "colorutil.h"
#include "triangles.h"
#include "split.h"
#include "tbb_config.h"
#include "CorrelationFunction.h"
#include "LocalQl.h"
#include "LocalQlNear.h"
#include "LocalWl.h"
#include "LocalWlNear.h"
#include "SolLiq.h"
#include "SolLiqNear.h"
#include "wigner3j.h"
#include "lindemann.h"
#include "LocalDescriptors.h"
#include "NearestNeighbors.h"

using namespace boost::python;
namespace bnp=boost::python::numeric;
using namespace freud;

/*! \file module.cc
    \brief _freud.so python exports
*/

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
void *my_import_array()
    {
    import_array();
    return NULL;
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
    density::export_CorrelationFunction();
    density::export_GaussianDensity();
    density::export_LocalDensity();
    voronoi::export_VoronoiBuffer();
    kspace::export_kspace();
    lindemann::export_lindemann();
    cluster::export_Cluster();
    cluster::export_ClusterProperties();
    order::export_HexOrderParameter();
    order::export_LocalDescriptors();
    locality::export_NearestNeighbors();
    interface::export_InterfaceMeasure();
    sphericalharmonicorderparameters::export_LocalQl();
    sphericalharmonicorderparameters::export_LocalQlNear();
    sphericalharmonicorderparameters::export_LocalWl();
    sphericalharmonicorderparameters::export_LocalWlNear();
    sphericalharmonicorderparameters::export_SolLiq();
    sphericalharmonicorderparameters::export_SolLiqNear();
    pairing::export_pairing();
    pmft::export_PMFXYZ();
    pmft::export_PMFXY2D();
    pmft::export_PMFTXYT2D();
    pmft::export_PMFTXYTP2D();
    pmft::export_PMFTXYTM2D();
    pmft::export_PMFTRPM();
    pmft::export_PMFTR12();
    shapesplit::export_ShapeSplit();
    bootstrap::export_Bootstrap();
    viz::export_colormap();
    viz::export_colorutil();
    viz::export_triangles();
    viz::export_split();
    parallel::export_tbb_config();
    }
