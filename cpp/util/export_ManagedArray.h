// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef EXPORT_MANAGED_ARRAY_H
#define EXPORT_MANAGED_ARRAY_H

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <string>

#include "ManagedArray.h"
#include "VectorMath.h"

namespace freud { namespace util { namespace detail {

template<typename T> nanobind::ndarray<nanobind::numpy, const T> toNumpyArray(nanobind::object self)
{
    ManagedArray<T>* self_cpp = nanobind::cast<ManagedArray<T>*>(self);
    auto dims = self_cpp->shape();
    auto ndim = dims.size();
    auto data_ptr = self_cpp->get();
    return nanobind::ndarray<nanobind::numpy, const T>((void*) data_ptr, ndim, &dims[0], self);
}

/* Need to alter array dimensions when returning an array of vec3*/
template<typename T> nanobind::ndarray<nanobind::numpy, const T> toNumpyArrayVec3(nanobind::object self)
{
    ManagedArray<vec3<T>>* self_cpp = nanobind::cast<ManagedArray<vec3<T>>*>(self);

    // get array data like before
    auto dims = self_cpp->shape();
    auto ndim = dims.size();
    auto data_ptr = self_cpp->get();

    // update the dimensions so it gets exposed to python the right way
    dims.push_back(3);
    ndim++;

    // now return the array
    return nanobind::ndarray<nanobind::numpy, const T>((void*) data_ptr, ndim, &dims[0], self);
}

template<typename T> void export_ManagedArray(nanobind::module_& m, const std::string& name)
{
    nanobind::class_<ManagedArray<T>>(m, name.c_str()).def("toNumpyArray", &toNumpyArray<T>);
}

template<typename T> void export_ManagedArrayVec3(nanobind::module_& m, const std::string& name)
{
    nanobind::class_<ManagedArray<vec3<T>>>(m, name.c_str()).def("toNumpyArray", &toNumpyArrayVec3<T>);
}

}; }; }; // namespace freud::util::detail

#endif
