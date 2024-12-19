#include <nanobind/nanobind.h>
#include <nanobind/nb_defs.h>

namespace freud::density::detail {

void export_CorrelationFunction(nanobind::module_& m);
void export_GaussianDensity(nanobind::module_& m);
void export_RDF(nanobind::module_& m);

}

using namespace freud::density::detail;

NB_MODULE(_density, module) // NOLINT(misc-use-anonymous-namespace): caused by nanobind
{
    export_RDF(module);
    export_GaussianDensity(module);
    export_CorrelationFunction(module);
}
