// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef EXPORT_NEIGHBOR_LIST_H
#define EXPORT_NEIGHBOR_LIST_H

#include "NeighborBond.h"
#include "NeighborList.h"

#include <nanobind/ndarray.h>

namespace freud { namespace locality {

namespace wrap {

template<typename T, typename shape = nanobind::shape<-1, 3>>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

void ConstructFromArrays(NeighborList* nlist, nb_array<unsigned int, nanobind::ndim<1>> query_point_indices,
                         unsigned int num_query_points,
                         nb_array<unsigned int, nanobind::ndim<1>> point_indices, unsigned int num_points,
                         nb_array<float> vectors, nb_array<float, nanobind::ndim<1>> weights);

void ConstructAllPairs(NeighborList* nlist, nb_array<float> points, nb_array<float> query_points,
                       const box::Box& box, const bool exclude_ii);

unsigned int filter(std::shared_ptr<NeighborList> nlist, nb_array<bool, nanobind::ndim<1>> filter);

}; // end namespace wrap

namespace detail {

void export_NeighborList(nanobind::module_& m);

void export_NeighborBond(nanobind::module_& m);

}; // namespace detail

}; }; // namespace freud::locality

#endif
