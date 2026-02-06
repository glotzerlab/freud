// Copyright (c) 2010-2026 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly

#include "BondHistogramCompute.h"
#include "NeighborList.h"
#include "NeighborQuery.h"
#include "RDF.h"
#include "VectorMath.h"

namespace freud { namespace density {

template<typename T, typename shape>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

namespace wrap {

void accumulateRDF(const std::shared_ptr<RDF>& self, const std::shared_ptr<locality::NeighborQuery>& nq,
                   const nb_array<const float, nanobind::shape<-1, 3>>& query_points,
                   const std::shared_ptr<locality::NeighborList>& nlist, const locality::QueryArgs& qargs)
{
    unsigned int const num_query_points = query_points.shape(0);
    const auto* query_points_data = reinterpret_cast<const vec3<float>*>(query_points.data());
    self->accumulate(nq, query_points_data, num_query_points, nlist, qargs);
}

}; // namespace wrap

namespace detail {

void export_RDF(nanobind::module_& m)
{
    nanobind::enum_<NormalizationMode>(m, "NormalizationMode")
        .value("exact", NormalizationMode::exact)
        .value("finite_size", NormalizationMode::finite_size);
    nanobind::class_<RDF, locality::BondHistogramCompute>(m, "RDF")
        .def(nanobind::init<unsigned int, float, float>())
        .def("accumulateRDF", &wrap::accumulateRDF, nanobind::arg("nq"), nanobind::arg("query_points"),
             nanobind::arg("nlist").none(), nanobind::arg("qargs"))
        .def("getRDF", &RDF::getRDF)
        .def("getNr", &RDF::getNr)
        .def_rw("mode", &RDF::mode);
}

} // namespace detail

}; }; // end namespace freud::density
