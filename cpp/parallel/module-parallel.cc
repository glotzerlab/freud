
#include <pybind11/pybind11.h>

#include "tbb_config.h"

using namespace freud::parallel;

PYBIND11_MODULE(_parallel, m)
{
    m.def("setNumThreads", &setNumThreads);
}
