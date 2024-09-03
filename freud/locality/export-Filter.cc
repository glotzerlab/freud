// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>  // NOLINT(misc-include-cleaner): used implicitly

#include "FilterRAD.h"
#include "FilterSANN.h"
#include "NeighborQuery.h"

namespace nb = nanobind;

namespace freud { namespace locality {

template<typename T, typename shape>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

namespace wrap {

void compute(std::shared_ptr<Filter> filter, std::shared_ptr<NeighborQuery> nq,
             nb_array<float, nb::shape<-1, 3>> query_points, std::shared_ptr<NeighborList> nlist,
             const QueryArgs& qargs)
{
    const auto num_query_points = query_points.shape(0);
    const auto* query_points_data = (vec3<float>*) query_points.data();
    filter->compute(nq, query_points_data, num_query_points, nlist, qargs);
}

}; // namespace wrap

namespace detail {

void export_Filter(nb::module_& m)
{
    nb::class_<Filter>(m, "Filter")
        .def("compute", &wrap::compute, nb::arg("nq"), nb::arg("query_points"), nb::arg("nlist").none(),
             nb::arg("qargs"))
        .def("getFilteredNlist", &Filter::getFilteredNlist)
        .def("getUnfilteredNlist", &Filter::getUnfilteredNlist);
}

void export_FilterRAD(nb::module_& m)
{
    nb::class_<FilterRAD, Filter>(m, "FilterRAD").def(nb::init<bool, bool>());
}

void export_FilterSANN(nb::module_& m)
{
    nb::class_<FilterSANN, Filter>(m, "FilterSANN").def(nb::init<bool>());
}

}; // namespace detail

}; }; // namespace freud::locality
