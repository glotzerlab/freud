#include <nanobind/nanobind.h>

#include "export_PMFT.h"

using namespace freud::pmft::detail;

NB_MODULE(_pmft, m)
{
    export_PMFT(m);
    //export_PMFTR12(m);
    //export_PMFTXY(m);
    //export_PMFTXYT(m);
    //export_PMFTXYZ(m);
}
