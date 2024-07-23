// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include "export_BondHistogramCompute.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/shared_ptr.h>

namespace nb = nanobind;

namespace freud { namespace locality {

namespace wrap {

/**
 * Here we convert a vector of vectors into a list of lists for returning to python.
 * */
template<typename T>
inline nb::object vectorVectorsToListLists(const std::vector<std::vector<T>>& vectorOfVectors)
{

    nb::list outer_python_list;
    for (const auto& vector : vectorOfVectors)
    {
        nb::list inner_python_list;
        for (const auto& element : vector)
        {
            inner_python_list.append(element);
        }
        outer_python_list.append(inner_python_list);
    }
    return outer_python_list;
}

nb::object getBinCenters(std::shared_ptr<BondHistogramCompute> bondHist)
{
    auto bin_centers_cpp = bondHist->getBinCenters();
    return vectorVectorsToListLists(bin_centers_cpp);
}

nb::object getBinEdges(std::shared_ptr<BondHistogramCompute> bondHist)
{
    auto bin_edges_cpp = bondHist->getBinEdges();
    return vectorVectorsToListLists(bin_edges_cpp);
}

nb::object getBounds(std::shared_ptr<BondHistogramCompute> bondHist)
{
    auto bounds_cpp = bondHist->getBounds();

    // convert the vector of pairs to a list of tuples
    nb::list python_list;
    for (const auto& pair : bounds_cpp)
    {
        nb::tuple python_tuple = nb::make_tuple(pair.first, pair.second);
        python_list.append(python_tuple);
    }
    return python_list;
}

}; // namespace wrap

namespace detail {

void export_BondHistogramCompute(nb::module_& m)
{
    nb::class_<BondHistogramCompute>(m, "BondHistogramCompute")
        .def("getBox", &BondHistogramCompute::getBox)
        .def("reset", &BondHistogramCompute::reset)
        .def("getBinCounts", &BondHistogramCompute::getBinCounts)
        .def("getAxisSizes", &BondHistogramCompute::getAxisSizes)
        .def("getBinCenters", &wrap::getBinCenters)
        .def("getBinEdges", &wrap::getBinEdges)
        .def("getBounds", &wrap::getBounds);
}

}; // namespace detail

}; }; // namespace freud::locality
