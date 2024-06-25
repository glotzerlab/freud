// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef EXPORT_VORONOI_H
#define EXPORT_VORONOI_H

#include "Voronoi.h"

#include <nanobind/nanobind.h>

namespace freud { namespace locality {

namespace wrap {

nanobind::object getPolytopes(std::shared_ptr<Voronoi> voro);

};

namespace detail {

void export_Voronoi(nanobind::module_& m);

};  // namespace detail

}; };  // namespace freud::locality

#endif
