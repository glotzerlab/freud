// Copyright (c) 2010-2025 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly

#include <utility>

#include "Filter.h"
#include "FilterRAD.h"
#include "FilterSANN.h"
#include "NeighborList.h"
#include "NeighborQuery.h"
#include "VectorMath.h"

namespace nb = nanobind;

namespace freud { namespace locality {

template<typename T, typename shape>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

namespace wrap {

void compute(const std::shared_ptr<Filter>& filter, std::shared_ptr<NeighborQuery> nq,
             const nb_array<float, nb::shape<-1, 3>>& query_points, std::shared_ptr<NeighborList> nlist,
             const QueryArgs& qargs)
{
    const auto num_query_points = query_points.shape(0);
    const auto* query_points_data = (vec3<float>*) query_points.data();
    filter->compute(std::move(nq), query_points_data, num_query_points, std::move(nlist), qargs);
}

}; // namespace wrap

namespace detail {

void export_Filter(nb::module_& module)
{
    nb::class_<Filter>(module, "Filter")
        .def("compute", &wrap::compute, nb::arg("nq"), nb::arg("query_points"), nb::arg("nlist").none(),
             nb::arg("qargs"))
        .def("getFilteredNlist", &Filter::getFilteredNlist)
        .def("getUnfilteredNlist", &Filter::getUnfilteredNlist);
}

void export_FilterRAD(nb::module_& module)
{
    nb::class_<FilterRAD, Filter>(module, "FilterRAD").def(nb::init<bool, bool>());
}

void export_FilterSANN(nb::module_& module)
{
    nb::class_<FilterSANN, Filter>(module, "FilterSANN").def(nb::init<bool>());
}

}; // namespace detail

}; }; // namespace freud::locality
