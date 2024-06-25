// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef EXPORT_BOX_H
#define EXPORT_BOX_H

#include "Box.h"

#include <nanobind/ndarray.h>

namespace freud { namespace box { namespace wrap {

template<typename T, typename shape = nanobind::shape<-1, 3>>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

void makeAbsolute(std::shared_ptr<Box> box, nb_array<float> vecs,
                  nb_array<float> out);

void makeFractional(std::shared_ptr<Box> box, nb_array<float> vecs,
                    nb_array<float> out);

void getImages(std::shared_ptr<Box> box, nb_array<float> vecs,
               nb_array<int> images);

void wrap(std::shared_ptr<Box> box, nb_array<float> vecs,
          nb_array<float> out);

void unwrap(std::shared_ptr<Box> box, nb_array<float> vecs,
        nb_array<int> images, nb_array<float> out);

std::vector<float> centerOfMass(std::shared_ptr<Box> box, nb_array<float> vecs,
                                nb_array<float, nanobind::shape<-1>> masses);

void center(std::shared_ptr<Box> box, nb_array<float> vecs,
        nb_array<float, nanobind::ndim<1>> masses);

void computeDistances(std::shared_ptr<Box> box, nb_array<float> query_points,
        nb_array<float> points, nb_array<float, nanobind::ndim<1>> distances);

void computeAllDistances(std::shared_ptr<Box> box, nb_array<float> query_points,
        nb_array<float> points, nb_array<float, nanobind::ndim<2>> distances);

void contains(std::shared_ptr<Box> box, nb_array<float> points,
              nb_array<bool, nanobind::ndim<1>> contains_mask);

}; }; }; // namespace freud::box::wrap

#endif
