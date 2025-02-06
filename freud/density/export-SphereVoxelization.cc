// Copyright (c) 2010-2025 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly

#include <utility>

#include "SphereVoxelization.h"

namespace freud { namespace density {

namespace wrap {
std::shared_ptr<SphereVoxelization> make_sphere_voxelization(unsigned int width_x, unsigned int width_y,
                                                             unsigned int width_z, float r_max)
{
    return std::make_shared<SphereVoxelization>(vec3<unsigned int>(width_x, width_y, width_z), r_max);
}

nanobind::tuple get_width(std::shared_ptr<SphereVoxelization> self)
{
    auto width = self->getWidth();
    return nanobind::make_tuple(width.x, width.y, width.z);
}
} // namespace wrap

namespace detail {

void export_SphereVoxelization(nanobind::module_& m)
{
    m.def("make_sphere_voxelization", &wrap::make_sphere_voxelization);
    nanobind::class_<SphereVoxelization>(m, "SphereVoxelization")
        .def("compute", &SphereVoxelization::compute, nanobind::arg("points"))
        .def("getWidth", &wrap::get_width)
        .def("getRMax", &SphereVoxelization::getRMax)
        .def_prop_ro("box", &SphereVoxelization::getBox)
        .def_prop_ro("voxels", &SphereVoxelization::getVoxels);
}

} // namespace detail

}; }; // namespace freud::density
