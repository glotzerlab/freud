#include <nanobind/nanobind.h>
#include <nanobind/nb_defs.h>

namespace freud::order::detail {
  // void export_PMFT(nanobind::module_& m);
  // void export_PMFTXY(nanobind::module_& m);
  void export_Nematic(nanobind::module_& m);
} // namespace freud::pmft::detail

using namespace freud::order::detail;

NB_MODULE(_order, module) // NOLINT(misc-use-anonymous-namespace): caused by nanobind
{
    export_Nematic(module);
}
