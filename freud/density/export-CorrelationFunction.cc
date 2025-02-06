// Copyright (c) 2010-2025 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include "CorrelationFunction.h"
#include "NeighborList.h"
#include "NeighborQuery.h"
#include "VectorMath.h"

namespace freud { namespace density {

template<typename T, typename shape>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

namespace wrap {

// Wrapper function for accumulate
void accumulateCF(const std::shared_ptr<CorrelationFunction>& self,
                  const std::shared_ptr<locality::NeighborQuery> neighbor_query,
                  const nb_array<std::complex<double>, nanobind::shape<-1>>& values,
                  const nb_array<float, nanobind::shape<-1, 3>>& query_points,
                  const nb_array<std::complex<double>, nanobind::shape<-1>>& query_values,
                  std::shared_ptr<locality::NeighborList> nlist, const locality::QueryArgs& qargs)
{
    auto* values_data = reinterpret_cast<std::complex<double>*>(values.data());
    auto* query_points_data = reinterpret_cast<vec3<float>*>(query_points.data());
    auto* query_values_data = reinterpret_cast<std::complex<double>*>(query_values.data());

    const unsigned int num_query_points = query_points.shape(0);

    self->accumulate(neighbor_query, values_data, query_points_data, query_values_data, num_query_points,
                     nlist, qargs);
}

} // namespace wrap

namespace detail {

void export_CorrelationFunction(nanobind::module_& m)
{
    nanobind::class_<CorrelationFunction>(m, "CorrelationFunction")
        .def(nanobind::init<unsigned int, float>(), nanobind::arg("bins"), nanobind::arg("r_max"))
        .def("reset", &CorrelationFunction::reset)
        .def("accumulate", &wrap::accumulateCF, nanobind::arg("neighbor_query"), nanobind::arg("values"),
             nanobind::arg("query_points"), nanobind::arg("query_values"), nanobind::arg("nlist").none(),
             nanobind::arg("qargs"))
        .def("getBinCenters", &CorrelationFunction::getBinCenters)
        .def("getAxisSizes", &CorrelationFunction::getAxisSizes)
        .def("getBinCounts", &CorrelationFunction::getBinCounts)
        .def("getBinEdges", &CorrelationFunction::getBinEdges)
        .def("getBox", &CorrelationFunction::getBox)
        .def("getCorrelation", &CorrelationFunction::getCorrelation);
}

} // namespace detail

}; }; // end namespace freud::density
