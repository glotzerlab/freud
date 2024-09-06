// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly

#include <utility>

#include "BondHistogramCompute.h"
#include "NeighborList.h"
#include "NeighborQuery.h"
#include "PMFT.h"
#include "PMFTXY.h"
#include "VectorMath.h"

namespace freud { namespace pmft {

template<typename T, typename shape>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

namespace wrap {

void accumulateXY(const std::shared_ptr<PMFTXY>& self, const std::shared_ptr<locality::NeighborQuery>& nq,
                  const nb_array<float, nanobind::shape<-1>>& query_orientations,
                  const nb_array<float, nanobind::shape<-1, 3>>& query_points,
                  std::shared_ptr<locality::NeighborList> nlist, const locality::QueryArgs& qargs)
{
    unsigned int const num_query_points = query_points.shape(0);
    auto* query_orientations_data = reinterpret_cast<float*>(query_orientations.data());
    auto* query_points_data = reinterpret_cast<vec3<float>*>(query_points.data());
    self->accumulate(nq, query_orientations_data, query_points_data, num_query_points, std::move(nlist),
                     qargs);
}

}; // namespace wrap

namespace detail {

void export_PMFT(nanobind::module_& m)
{
    nanobind::class_<PMFT, locality::BondHistogramCompute>(m, "PMFT").def("getPCF", &PMFT::getPCF);
}

void export_PMFTXY(nanobind::module_& m)
{
    nanobind::class_<PMFTXY, PMFT>(m, "PMFTXY")
        .def(nanobind::init<float, float, unsigned int, unsigned int>())
        .def("accumulate", &wrap::accumulateXY, nanobind::arg("nq"), nanobind::arg("query_orientations"),
             nanobind::arg("query_points"), nanobind::arg("nlist").none(), nanobind::arg("qargs"));
}

} // namespace detail

}; }; // end namespace freud::pmft
