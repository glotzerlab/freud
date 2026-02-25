// Copyright (c) 2010-2026 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly

#include "NeighborQuery.h"
#include "StaticStructureFactor.h"
#include "VectorMath.h"

namespace freud::diffraction {

template<typename T, typename shape>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

namespace wrap {

void accumulate(const std::shared_ptr<StaticStructureFactor>& self,
                const std::shared_ptr<locality::NeighborQuery>& neighbor_query,
                const nb_array<const float, nanobind::shape<-1, 3>>& query_points, unsigned int n_total)
{
    const auto* query_points_data = reinterpret_cast<const vec3<float>*>(query_points.data());
    const unsigned int n_query_points = query_points.shape(0);
    self->accumulate(neighbor_query, query_points_data, n_query_points, n_total);
};

} // namespace wrap

namespace detail {

void export_StaticStructureFactor(nanobind::module_& m)
{
    nanobind::class_<StaticStructureFactor>(m, "StaticStructureFactor")
        .def("accumulate", &wrap::accumulate, nanobind::arg("neighbor_query"), nanobind::arg("query_points"),
             nanobind::arg("n_total"))
        .def("reset", &StaticStructureFactor::reset)
        .def("getStructureFactor", &StaticStructureFactor::getStructureFactor)
        .def("getMinValidK", &StaticStructureFactor::getMinValidK)
        .def("getBinCenters", &StaticStructureFactor::getBinCenters)
        .def("getBinEdges", &StaticStructureFactor::getBinEdges);
}

} // namespace detail
} // namespace freud::diffraction
