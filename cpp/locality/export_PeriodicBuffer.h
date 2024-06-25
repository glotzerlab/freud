// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef EXPORT_PERIODIC_BUFFER_H
#define EXPORT_PERIODIC_BUFFER_H

#include "PeriodicBuffer.h"

#include <nanobind/nanobind.h>

namespace freud { namespace locality {

namespace detail {

void export_PeriodicBuffer(nanobind::module_& m);

};  // namespace detail

}; };  // namespace freud::locality

#endif
