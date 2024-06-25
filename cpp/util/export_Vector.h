// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef EXPORT_VECTOR_H
#define EXPORT_VECTOR_H

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <string>
#include <vector>

#include "VectorMath.h"

namespace freud { namespace util {

namespace wrap {

template<typename T> nanobind::ndarray<nanobind::numpy, const T> vectorToNumpyArray(nanobind::object self)
{
    std::vector<T>* self_cpp = nanobind::cast<std::vector<T>*>(self);
    auto dims = {self_cpp->size()};
    auto ndim = 1;
    auto data_ptr = self_cpp->data();
    return nanobind::ndarray<nanobind::numpy, const T>((void*) data_ptr, dims, self);
}

/* Need to alter array dimensions when returning an array of vec3*/
template<typename T> nanobind::ndarray<nanobind::numpy, const T> vectorToNumpyArrayVec3(nanobind::object self)
{
    std::vector<vec3<T>>* self_cpp = nanobind::cast<std::vector<vec3<T>>*>(self);

    // get array data like before
    size_t size = self_cpp->size();
    std::initializer_list<size_t> dims = {size, 3};
    auto ndim = 2;
    auto data_ptr = self_cpp->data();

    // now return the array
    return nanobind::ndarray<nanobind::numpy, const T>((void*) data_ptr, dims, self);
}

}; // namespace wrap

namespace detail {

template<typename T> void export_Vector(nanobind::module_& m, const std::string& name)
{
    nanobind::class_<std::vector<T>>(m, name.c_str()).def("toNumpyArray", &wrap::vectorToNumpyArray<T>);
}

template<typename T> void export_VectorVec3(nanobind::module_& m, const std::string& name)
{
    nanobind::class_<std::vector<vec3<T>>>(m, name.c_str())
        .def("toNumpyArray", &wrap::vectorToNumpyArrayVec3<T>);
}

}; // namespace detail

}; }; // namespace freud::util

#endif
