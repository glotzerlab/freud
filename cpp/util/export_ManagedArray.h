#ifndef EXPORT_MANAGED_ARRAY_H
#define EXPORT_MANAGED_ARRAY_H

#include <nanobind/nanobind.h>

template<typename T>
void export_ManagedArray(nb::module_& m, const std::string& name)
{
    nb::class_<ManagedArray<T>>(m, name)
        .def("toNumpyArray", ()[]{});
}

#endif
