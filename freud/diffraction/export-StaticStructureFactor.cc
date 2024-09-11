// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>

#include "StaticStructureFactor.h"

namespace freud { namespace diffraction {

    template<typename T, typename shape>
    using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

    namespace wrap {

        void accumulate(std::shared_ptr<StaticStructureFactor> self,
                        std::shared_ptr<locality::NeighborQuery> neighbor_query,
                        // const vec3<float>* query_points,
                        nb_array<float, nanobind::shape<-1, 3>> query_points,
                        unsigned int n_query_points,
                        unsigned int n_total)
        {
            auto* query_points_data = reinterpret_cast<vec3<float>*>(query_points.data());
            self->accumulate(neighbor_query, query_points_data, n_query_points, n_total);
        };

    } // namespace wrap

    namespace detail {

        void export_StaticStructureFactor(nanobind::module_& m)
        {
            nanobind::class_<StaticStructureFactor>(m, "StaticStructureFactor")
                // .def(nanobind::init<unsigned int, float, float>())
                .def("accumulate", &wrap::accumulate, nanobind::arg("neighbor_query"),
                    nanobind::arg("query_points"), nanobind::arg("n_query_points"),
                    nanobind::arg("n_total"))
                .def("reset", &StaticStructureFactor::reset)
                .def("getStructureFactor", &StaticStructureFactor::getStructureFactor)
                .def("getMinValidK", &StaticStructureFactor::getMinValidK);
        }

    } // namespace detail
}}