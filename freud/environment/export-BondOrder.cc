// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly
#include <nanobind/stl/vector.h>     // NOLINT(misc-include-cleaner): used implicitly
#include <vector>

#include "BondOrder.h"

namespace nb = nanobind;

namespace freud { namespace environment {

template<typename T, typename shape> using nb_array = nb::ndarray<T, shape, nb::device::cpu, nb::c_contig>;

namespace wrap {

void accumulateBondOrder(const std::shared_ptr<BondOrder>& self,
                         const std::shared_ptr<locality::NeighborQuery>& nq,
                         const nb_array<float, nb::shape<-1, 4>>& orientations,
                         const nb_array<float, nb::shape<-1, 3>>& query_points,
                         const nb_array<float, nb::shape<-1, 4>>& query_orientations,
                         std::shared_ptr<locality::NeighborList> nlist, const locality::QueryArgs& qargs)
{
    unsigned int const n_query_points = query_points.shape(0);
    // std::cout << n_query_points << std::endl;

    // if (query_points.is_none()){
    //   auto* query_points_data = nq->getPoints();
    // }
    // else {
    //   auto* query_points_data= reinterpret_cast<vec3<float>*>(query_points.data());
    // }

    auto* orientations_data = reinterpret_cast<quat<float>*>(orientations.data());
    auto* query_points_data = reinterpret_cast<vec3<float>*>(query_points.data());
    auto* query_orientations_data = reinterpret_cast<quat<float>*>(query_orientations.data());

    self->accumulate(nq, orientations_data, query_points_data, query_orientations_data, n_query_points, nlist,
                     qargs);
}

}; // namespace wrap

namespace detail {

void export_BondOrder(nb::module_& module)
{
    nb::enum_<BondOrderMode>(module, "BondOrderMode")
        .value("bod", BondOrderMode::bod)
        .value("lbod", BondOrderMode::lbod)
        .value("obcd", BondOrderMode::obcd)
        .value("oocd", BondOrderMode::oocd)
        .export_values();

    nb::class_<BondOrder>(module, "BondOrder")
        .def(nb::init<unsigned int, unsigned int, BondOrderMode>())
        .def("getBondOrder", &BondOrder::getBondOrder)
        .def("getBinCounts", &BondOrder::getBinCounts)
        .def("getBinCenters", &BondOrder::getBinCenters)
        .def("getBinEdges", &BondOrder::getBinEdges)
        .def("getBox", &BondOrder::getBox)
        .def("getAxisSizes", &BondOrder::getAxisSizes)
        .def("getMode", &BondOrder::getMode)
        .def("accumulate", &wrap::accumulateBondOrder, nanobind::arg("nq").none(),
             nanobind::arg("orientations"), nanobind::arg("query_points"),
             nanobind::arg("query_orientations"),
             // nanobind::arg("n_query_points"),
             nanobind::arg("nlist").none(), nanobind::arg("qargs").none())
        .def("reset", &BondOrder::reset);
}

}; // namespace detail

}; }; // namespace freud::environment
