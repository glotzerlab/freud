#ifndef EXPORT_PMFT_H
#define EXPORT_PMFT_H

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "PMFT.h"
#include "BondHistogramCompute.h"

namespace freud { namespace pmft {

namespace detail
{

void export_PMFT(nanobind::module_& m)
{
    nanobind::class_<PMFT, locality::BondHistogramCompute>(m, "PMFT")
        .def("getPCF", &PMFT::getPCF);
}

}  // namespace detail

}; };  // end namespace freud::pmft

#endif
