#ifndef EXPORT_PMFT_H
#define EXPORT_PMFT_H

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>

#include "PMFT.h"
#include "PMFTXY.h"
#include "BondHistogramCompute.h"

namespace freud { namespace pmft {

template<typename T, typename shape>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

namespace wrap
{

void accumulateXY(std::shared_ptr<PMFTXY> pmftxy, std::shared_ptr<locality::NeighborQuery> nq,
        nb_array<float, nanobind::shape<-1>> query_orientations,
        nb_array<float, nanobind::shape<-1, 3>> query_points,
        std::shared_ptr<locality::NeighborList> nlist, const locality::QueryArgs& qargs)
{
    unsigned int num_query_points = query_points.shape(0);
    auto* query_orientations_data = reinterpret_cast<float*>(query_orientations.data());
    auto* query_points_data = reinterpret_cast<vec3<float>*>(query_points.data());
    pmftxy->accumulate(nq, query_orientations_data, query_points_data, num_query_points,
            nlist, qargs);
}

};

namespace detail
{

void export_PMFT(nanobind::module_& m)
{
    nanobind::class_<PMFT, locality::BondHistogramCompute>(m, "PMFT")
        .def("getPCF", &PMFT::getPCF);
}

void export_PMFTXY(nanobind::module_& m)
{
    nanobind::class_<PMFTXY, PMFT>(m, "PMFTXY")
        .def(nanobind::init<float, float, unsigned int, unsigned int>())
        .def("accumulate", &wrap::accumulateXY, nanobind::arg("nq"), nanobind::arg("query_orientations"), nanobind::arg("query_points"), nanobind::arg("nlist").none(), nanobind::arg("qargs"));
}

}  // namespace detail

}; };  // end namespace freud::pmft

#endif
