
#include <nanobind/nanobind.h>

#include "tbb_config.h"

using namespace freud::parallel;

NB_MODULE(_parallel, m)
{
    m.def("setNumThreads", &setNumThreads);
}
