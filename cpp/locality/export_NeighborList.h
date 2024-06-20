#ifndef EXPORT_NEIGHBOR_LIST_H
#define EXPORT_NEIGHBOR_LIST_H

#include "NeighborList.h"
#include "NeighborBond.h"

#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
namespace nb = nanobind;


namespace freud { namespace locality {

//template<typename T, typename shape = nb::shape<-1, 3>>
//using nb_array = nb::ndarray<T, shape, nb::device::cpu, nb::c_contig>;

namespace wrap
{

std::shared_ptr<NeighborList> fromNumBonds(unsigned int num_bonds)
{
    return std::make_shared<NeighborList>(num_bonds);
}

std::shared_ptr<NeighborList> fromArrays(nb_array<unsigned int, nb::ndim<1>> query_point_indices,
        unsigned int num_query_points, nb_array<unsigned int, nb::ndim<1>> point_indices,
        unsigned int num_points, nb_array<float> vectors, nb_array<float, nb::ndim<1>> weights)
{
    const unsigned int num_bonds = query_point_indices.shape(0);
    const auto *query_point_indices_data = (const unsigned int *)query_point_indices.data();
    const auto *point_indices_data = (const unsigned int *)point_indices.data();
    const auto *vectors_data = (const vec3<float>*)vectors.data();
    const auto *weights_data = (const float*)weights.data();
    return std::make_shared<NeighborList>(num_bonds, query_point_indices_data,
            num_query_points, point_indices_data, num_points, vectors_data, weights_data);
}

std::shared_ptr<NeighborList> allPairs(nb_array<float> points, nb_array<float> query_points,
        const box::Box& box, const bool exclude_ii)
{
    const unsigned int num_points = points.shape(0);
    const auto *points_data = (const vec3<float> *)points.data();
    const unsigned int num_query_points = query_points.shape(0);
    const auto *query_points_data = (const vec3<float> *)query_points.data();
    return std::make_shared<NeighborList>(points_data, query_points_data, box,
            exclude_ii, num_points, num_query_points);
}

unsigned int filter(std::shared_ptr<NeighborList> nlist, nb_array<bool, nb::ndim<1>> filter)
{
    const bool* filter_data = (const bool *) filter.data();
    return nlist->filter(filter_data);
}

nb::ndarray<nb::numpy, float, nb::ndim<1>> getDistances(std::shared_ptr<NeighborList> nlist)
{
    // get the array info
    const unsigned int num_bonds = nlist->getNumBonds();
    const auto& distances = nlist->getDistances();
    const float *distances_ptr = &distances(0);

    // create a python object to tie the lifetime of the array to
    nb::handle py_class = nb::type<NeighborList>();
    nb::object py_instance = nb::inst_alloc(py_class);
    nb::inst_zero(py_instance);

    // return an array with the lifetime tied to the python instance
    return nb::ndarray<nb::numpy, float, nb::ndim<1>>(
        (void *)distances_ptr, { num_bonds }, py_instance
    );
}

};  // end namespace wrap

namespace detail
{
void export_NeighborList(nb::module_& m)
{
    nb::class_<NeighborList>(m, "NeighborList")
        // export constructors, wrap some as static factory methods
        .def(nb::init<>())
        .def_static("fromNumBonds", &wrap::fromNumBonds)
        .def_static("fromArrays", &wrap::fromArrays)
        .def_static("allPairs", &wrap::allPairs)
        // getters and setters
        //.def("getNeighbors")
        //.def("getWeights")
        .def("getDistances", &wrap::getDistances)
        //.def("getVectors")
        //.def("getSegments")
        //.def("getCounts")
        .def("getNumBonds", &NeighborList::getNumBonds)
        .def("getNumQueryPoints", &NeighborList::getNumQueryPoints)
        .def("getNumPoints", &NeighborList::getNumPoints)
        // other member functions
        .def("copy", &NeighborList::copy)
        .def("find_first_index", &NeighborList::find_first_index)
        .def("filter", &wrap::filter)
        .def("filter_r", &NeighborList::filter_r)
        .def("sort", &NeighborList::sort);
};

void export_NeighborBond(nb::module_& m)
{
    nb::class_<NeighborBond>(m, "NeighborBond")
        .def(nb::init<>())
        .def("getQueryPointIdx", &NeighborBond::getQueryPointIdx)
        .def("getPointIdx", &NeighborBond::getPointIdx)
        .def("getDistance", &NeighborBond::getDistance)
        .def(nb::self == nb::self)
        .def(nb::self != nb::self);
};
};

}; };  // end namespace freud:locality

#endif
