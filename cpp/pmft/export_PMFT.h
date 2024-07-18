#ifndef EXPORT_PMFT_H
#define EXPORT_PMFT_H

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "PMFT.h"
#include "BondHistogramCompute.h"

namespace freud { namespace pmft {

namespace detail
{

void export_PMFT(nanobind::module& m, const std::string& name)
{
    nanobind::class_<PMFT, BondHistogramCompute>(m, name)
        .def("getPCF", &PMFT::getPCF);
}

}  // namespace detail

}; };  // end namespace freud::pmft

#endif
