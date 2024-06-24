// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef EXPORT_VORONOI_H
#define EXPORT_VORONOI_H

#include "Voronoi.h"

#include <nanobind/ndarray.h>
namespace nb = nanobind;

namespace freud { namespace locality {

namespace wrap
{

// TODO this is very inefficient, we are converting data both on the
// C++ and python side, it would be better to find an efficient way to pass
// ragged data back and forth that other modules could use as well
nb::object getPolytopes(std::shared_ptr<Voronoi> voro)
{
    // get cpp data
    auto polytopes_cpp = voro->getPolytopes();

    // convert to list of of list of lists for returning to python
    nb::list polytopes_python;
    for (const auto& polytope_cpp : polytopes_cpp)
    {
        nb::list polytope;
        for (const auto& vertex_cpp : polytope_cpp)
        {
            nb::list vertex;
            vertex.append(vertex_cpp.x);
            vertex.append(vertex_cpp.y);
            vertex.append(vertex_cpp.z);
            polytope.append(vertex);
        }
        polytopes_python.append(polytope);
    }
    return polytopes_python;
}

};

namespace detail {

void export_Voronoi(nb::module_& m)
{
    nb::class_<Voronoi>(m, "Voronoi")
        .def(nb::init<>())
        .def("compute", &Voronoi::compute)
        .def("getBox", &Voronoi::getBox)
        .def("getVolumes", &Voronoi::getVolumes)
        .def("getPolytopes", &wrap::getPolytopes)
        .def("getNeighborList", &Voronoi::getNeighborList);
};

}; }; };  // namespace freud::locality::detail

#endif
