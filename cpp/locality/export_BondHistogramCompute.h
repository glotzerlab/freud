// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef EXPORT_BOND_HISTOGRAM_COMPUTE_H
#define EXPORT_BOND_HISTOGRAM_COMPUTE_H

#include "BondHistogramCompute.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace freud { namespace locality {

namespace wrap {

nanobind::object getBinCenters(std::shared_ptr<BondHistogramCompute> bondHist);

nanobind::object getBinEdges(std::shared_ptr<BondHistogramCompute> bondHist);

nanobind::object getBounds(std::shared_ptr<BondHistogramCompute> bondHist);

nanobind::object getAxisSizes(std::shared_ptr<BondHistogramCompute> bondHist);

}; // namespace wrap

namespace detail {

void export_BondHistogramCompute(nanobind::module_& m);

}; // namespace detail

}; }; // namespace freud::locality

#endif
