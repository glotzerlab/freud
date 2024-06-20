
#include <nanobind/nanobind.h>

#include "export_ManagedArray.h"

using namespace freud::util

NB_MODULE(_util, m)
{
    export_ManagedArray<float>(m, "ManagedArray_float");
}
