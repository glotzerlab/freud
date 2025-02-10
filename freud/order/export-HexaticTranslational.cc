// Copyright (c) 2010-2025 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly

#include "HexaticTranslational.h"
#include "NeighborList.h"
#include "NeighborQuery.h"

namespace freud { namespace order {

template<typename T, typename shape>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

namespace wrap {

void computeHexaticTranslational(const std::shared_ptr<Hexatic>& self,
                                 const std::shared_ptr<locality::NeighborList>& nlist,
                                 std::shared_ptr<locality::NeighborQuery>& points,
                                 const locality::QueryArgs& qargs)
{
    self->compute(nlist, points, qargs);
}

}; // namespace wrap

namespace detail {

void export_HexaticTranslational(nanobind::module_& m)
{
    nanobind::class_<Hexatic>(m, "Hexatic")
        .def(nanobind::init<unsigned int, bool>())
        .def("compute", &wrap::computeHexaticTranslational, nanobind::arg("nlist").none(),
             nanobind::arg("points"), nanobind::arg("qargs"))
        .def("getK", &HexaticTranslational<unsigned int>::getK)
        .def("getOrder", &HexaticTranslational<unsigned int>::getOrder)
        .def("isWeighted", &HexaticTranslational<unsigned int>::isWeighted);
}

} // namespace detail

}; }; // namespace freud::order
