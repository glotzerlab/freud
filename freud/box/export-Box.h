// Copyright (c) 2010-2025 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef EXPORT_BOX_H
#define EXPORT_BOX_H

#include "Box.h"

#include <memory>
#include <nanobind/ndarray.h>
#include <vector>

namespace freud { namespace box { namespace wrap {

template<typename T, typename shape = nanobind::shape<-1, 3>>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

void makeAbsolute(const std::shared_ptr<Box>& box, const nb_array<const float>& vecs, const nb_array<float>& out);

void makeFractional(const std::shared_ptr<Box>& box, const nb_array<const float>& vecs, const nb_array<float>& out);

void getImages(const std::shared_ptr<Box>& box, const nb_array<const float>& vecs, const nb_array<int>& images);

void wrap(const std::shared_ptr<Box>& box, const nb_array<const float>& vecs, const nb_array<float>& out);

void unwrap(const std::shared_ptr<Box>& box, const nb_array<const float>& vecs, const nb_array<const int>& images,
            const nb_array<float>& out);

std::vector<float> centerOfMass(const std::shared_ptr<Box>& box, const nb_array<float>& vecs,
                                const nb_array<const float, nanobind::shape<-1>>& masses);

void center(const std::shared_ptr<Box>& box, const nb_array<float>& vecs,
            const nb_array<const float, nanobind::ndim<1>>& masses);

void computeDistances(const std::shared_ptr<Box>& box, const nb_array<const float>& query_points,
                      const nb_array<const float>& points, const nb_array<float, nanobind::ndim<1>>& distances);

void computeAllDistances(const std::shared_ptr<Box>& box, const nb_array<const float>& query_points,
                         const nb_array<const float>& points, const nb_array<float, nanobind::ndim<2>>& distances);

void contains(const std::shared_ptr<Box>& box, const nb_array<float>& points,
              const nb_array<bool, nanobind::ndim<1>>& contains_mask);

}; }; }; // namespace freud::box::wrap

#endif
