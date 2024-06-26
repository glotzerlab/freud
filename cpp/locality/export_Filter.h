// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef EXPORT_FILTER_H
#define EXPORT_FILTER_H

#include "Filter.h"
#include "FilterRAD.h"
#include "FilterSANN.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace freud { namespace locality {

namespace wrap {

template<typename T, typename shape>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

void compute(std::shared_ptr<Filter> filter, std::shared_ptr<NeighborQuery> nq, nb_array<float, nanobind::shape<-1, 3>> query_points,
             std::shared_ptr<NeighborList> nlist, const QueryArgs& qargs);

}; // namespace wrap

namespace detail {

void export_Filter(nanobind::module_& m);

void export_FilterRAD(nanobind::module_& m);

void export_FilterSANN(nanobind::module_& m);

}; // namespace detail

}; }; // namespace freud::locality

#endif
