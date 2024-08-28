// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>

#include "PMFTR12.h"

namespace freud { namespace pmft {

template<typename T, typename shape>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

namespace wrap {

void accumulateR12(std::shared_ptr<PMFTR12> self, std::shared_ptr<locality::NeighborQuery> nq,
                  nb_array<float, nanobind::shape<-1>> orientations,
                  nb_array<float, nanobind::shape<-1, 3>> query_points,
                  nb_array<float, nanobind::shape<-1>> query_orientations,
                  std::shared_ptr<locality::NeighborList> nlist, const locality::QueryArgs& qargs)
{
    unsigned int num_query_points = query_points.shape(0);
    auto* orientations_data = reinterpret_cast<float*>(orientations.data());
    auto* query_orientations_data = reinterpret_cast<float*>(query_orientations.data());
    auto* query_points_data = reinterpret_cast<vec3<float>*>(query_points.data());
    self->accumulate(nq, orientations_data,query_points_data, query_orientations_data,  num_query_points, nlist, qargs);
}

}; // namespace wrap

namespace detail {

void export_PMFTR12(nanobind::module_& m)
{
    nanobind::class_<PMFTR12, PMFT>(m, "PMFTR12")
        .def(nanobind::init<float, unsigned int, unsigned int, unsigned int>())
        .def("accumulate", &wrap::accumulateR12, nanobind::arg("nq"), nanobind::arg("orientations"), nanobind::arg("query_points"), nanobind::arg("query_orientations"),
             nanobind::arg("nlist").none(), nanobind::arg("qargs"));
}

} // namespace detail

}; }; // end namespace freud::pmft
