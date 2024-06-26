// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef EXPORT_NEIGHBOR_QUERY_H
#define EXPORT_NEIGHBOR_QUERY_H

#include "AABBQuery.h"
#include "LinkCell.h"
#include "NeighborQuery.h"
#include "RawPoints.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace freud { namespace locality {

namespace wrap {

template<typename T, typename shape>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

std::shared_ptr<NeighborQueryIterator> query(std::shared_ptr<NeighborQuery> nq, nb_array<float, nanobind::shape<-1, 3>> query_points,
                                             const QueryArgs& qargs);

void AABBQueryConstructor(AABBQuery* nq, const box::Box& box, nb_array<float, nanobind::shape<-1, 3>> points);

void LinkCellConstructor(LinkCell* nq, const box::Box& box, nb_array<float, nanobind::shape<-1, 3>> points, float cell_width);

void RawPointsConstructor(RawPoints* nq, const box::Box& box, nb_array<float, nanobind::shape<-1, 3>> points);

}; // namespace wrap

namespace detail {

void export_NeighborQuery(nanobind::module_& m);

void export_AABBQuery(nanobind::module_& m);

void export_LinkCell(nanobind::module_& m);

void export_RawPoints(nanobind::module_& m);

void export_QueryArgs(nanobind::module_& m);

void export_NeighborQueryIterator(nanobind::module_& m);

}; // namespace detail

}; }; // namespace freud::locality

#endif
