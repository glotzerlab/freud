// Copyright (c) 2010-2025 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly

#include <utility>

#include "NeighborList.h"
#include "NeighborQuery.h"
#include "PMFT.h"
#include "PMFTXYZ.h"
#include "VectorMath.h"

namespace freud { namespace pmft {

template<typename T, typename shape>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

namespace wrap {

void accumulateXYZ(const std::shared_ptr<PMFTXYZ>& self, const std::shared_ptr<locality::NeighborQuery>& nq,
                   const nb_array<const float, nanobind::shape<-1, 4>>& query_orientations,
                   const nb_array<const float, nanobind::shape<-1, 3>>& query_points,
                   const nb_array<const float, nanobind::shape<-1, 4>>& equivalent_orientations,
                   std::shared_ptr<locality::NeighborList> nlist, const locality::QueryArgs& qargs)
{
    unsigned int const num_query_points = query_points.shape(0);
    const auto* query_orientations_data = reinterpret_cast<const quat<float>*>(query_orientations.data());
    const auto* query_points_data = reinterpret_cast<const vec3<float>*>(query_points.data());
    const auto* equivalent_orientations_data
        = reinterpret_cast<const quat<float>*>(equivalent_orientations.data());
    unsigned int const num_equivalent_orientations = equivalent_orientations.shape(0);
    self->accumulate(nq, query_orientations_data, query_points_data, num_query_points,
                     equivalent_orientations_data, num_equivalent_orientations, std::move(nlist), qargs);
}

}; // namespace wrap

namespace detail {

void export_PMFTXYZ(nanobind::module_& m)
{
    nanobind::class_<PMFTXYZ, PMFT>(m, "PMFTXYZ")
        .def(nanobind::init<float, float, float, unsigned int, unsigned int, unsigned int>())
        .def("accumulate", &wrap::accumulateXYZ, nanobind::arg("nq"), nanobind::arg("query_orientations"),
             nanobind::arg("query_points"), nanobind::arg("equiv_orientations"),
             nanobind::arg("nlist").none(), nanobind::arg("qargs"));
}

} // namespace detail

}; }; // end namespace freud::pmft
