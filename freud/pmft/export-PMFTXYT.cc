// Copyright (c) 2010-2026 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly

#include <utility>

#include "NeighborList.h"
#include "NeighborQuery.h"
#include "PMFT.h"
#include "PMFTXYT.h"
#include "VectorMath.h"

namespace freud { namespace pmft {

template<typename T, typename shape>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

namespace wrap {

void accumulateR12(const std::shared_ptr<PMFTXYT>& self, const std::shared_ptr<locality::NeighborQuery>& nq,
                   const nb_array<const float, nanobind::shape<-1>>& orientations,
                   const nb_array<const float, nanobind::shape<-1, 3>>& query_points,
                   const nb_array<const float, nanobind::shape<-1>>& query_orientations,
                   std::shared_ptr<locality::NeighborList> nlist, const locality::QueryArgs& qargs)
{
    unsigned int const num_query_points = query_points.shape(0);
    const auto* orientations_data = reinterpret_cast<const float*>(orientations.data());
    const auto* query_orientations_data = reinterpret_cast<const float*>(query_orientations.data());
    const auto* query_points_data = reinterpret_cast<const vec3<float>*>(query_points.data());
    self->accumulate(nq, orientations_data, query_points_data, query_orientations_data, num_query_points,
                     std::move(nlist), qargs);
}

}; // namespace wrap

namespace detail {

void export_PMFTXYT(nanobind::module_& m)
{
    nanobind::class_<PMFTXYT, PMFT>(m, "PMFTXYT")
        .def(nanobind::init<float, float, unsigned int, unsigned int, unsigned int>())
        .def("accumulate", &wrap::accumulateR12, nanobind::arg("nq"), nanobind::arg("orientations"),
             nanobind::arg("query_points"), nanobind::arg("query_orientations"),
             nanobind::arg("nlist").none(), nanobind::arg("qargs"));
}

} // namespace detail

}; }; // end namespace freud::pmft
