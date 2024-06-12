// Copyright (c) 2010-2023 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef EXPORT_BOX_H
#define EXPORT_BOX_H

#include "Box.h"

#include <nanobind/ndarray.h>
namespace nb = nanobind;

namespace freud { namespace box {

template<typename T, typename shape = nb::shape<-1, 3>>
using nb_array = nb::ndarray<T, shape, nb::device::cpu, nb::c_contig>;

void makeAbsolutePython(Box box, nb_array<float, nb::shape<-1, 3>> vecs,
                        nb_array<float, nb::shape<-1, 3>> out)
{
    unsigned int Nvecs = vecs.shape(0);
    vec3<float>* vecs_data = (vec3<float>*) (vecs.data());
    vec3<float>* out_data = (vec3<float>*) (out.data());
    box.makeAbsolute(vecs_data, Nvecs, out_data);
}

void makeFractionalPython(Box box, nb_array<float, nb::shape<-1, 3>> vecs,
                          nb_array<float, nb::shape<-1, 3>> out)
{
    unsigned int Nvecs = vecs.shape(0);
    vec3<float>* vecs_data = (vec3<float>*) (vecs.data());
    vec3<float>* out_data = (vec3<float>*) (out.data());
    box.makeFractional(vecs_data, Nvecs, out_data);
}

void getImagesPython(Box box, nb_array<float, nb::shape<-1, 3>> vecs, nb_array<int, nb::shape<-1, 3>> images)
{
    const unsigned int Nvecs = vecs.shape(0);
    vec3<float>* vecs_data = (vec3<float>*) (vecs.data());
    vec3<int>* images_data = (vec3<int>*) (images.data());
    box.getImages(vecs_data, Nvecs, images_data);
}

void wrapPython(Box box, nb_array<float, nb::shape<-1, 3>> vecs, nb_array<float, nb::shape<-1, 3>> out)
{
    const unsigned int Nvecs = vecs.shape(0);
    vec3<float>* vecs_data = (vec3<float>*) (vecs.data());
    vec3<float>* out_data = (vec3<float>*) (out.data());
    box.wrap(vecs_data, Nvecs, out_data);
}

void unwrapPython(Box box, nb_array<float> vecs, nb_array<int> images, nb_array<float> out)
{
    const unsigned int Nvecs = vecs.shape(0);
    vec3<float>* vecs_data = (vec3<float>*) (vecs.data());
    vec3<int>* images_data = (vec3<int>*) (images.data());
    vec3<float>* out_data = (vec3<float>*) (out.data());
    box.unwrap(vecs_data, images_data, Nvecs, out_data);
}

std::vector<float> centerOfMassPython(Box box, nb_array<float> vecs, nb_array<float, nb::shape<-1>> masses)
{
    const unsigned int Nvecs = vecs.shape(0);
    vec3<float>* vecs_data = (vec3<float>*) (vecs.data());
    float* masses_data = (float*) (masses.data());
    auto com = box.centerOfMass(vecs_data, Nvecs, masses_data);
    return {com.x, com.y, com.z};
}

void centerPython(Box box, nb_array<float> vecs, nb_array<float, nb::ndim<1>> masses)
{
    const unsigned int Nvecs = vecs.shape(0);
    vec3<float>* vecs_data = (vec3<float>*) (vecs.data());
    float* masses_data = (float*) (masses.data());
    box.center(vecs_data, Nvecs, masses_data);
}

void computeDistancesPython(Box box, nb_array<float> query_points, nb_array<float> points,
                            nb_array<float, nb::ndim<1>> distances)
{
    const unsigned int n_query_points = query_points.shape(0);
    vec3<float>* query_points_data = (vec3<float>*) (query_points.data());
    const unsigned int n_points = points.shape(0);
    vec3<float>* points_data = (vec3<float>*) (points.data());
    float* distances_data = (float*) (distances.data());
    if (n_query_points != n_points)
    {
        throw std::invalid_argument("The number of query points and points must match.");
    }
    box.computeDistances(query_points_data, n_query_points, points_data, n_points, distances_data);
}

void computeAllDistancesPython(Box box, nb_array<float> query_points, nb_array<float> points,
                               nb_array<float, nb::ndim<2>> distances)
{
    const unsigned int n_query_points = query_points.shape(0);
    vec3<float>* query_points_data = (vec3<float>*) (query_points.data());
    const unsigned int n_points = points.shape(0);
    vec3<float>* points_data = (vec3<float>*) (points.data());
    float* distances_data = (float*) (distances.data());
    box.computeAllDistances(query_points_data, n_query_points, points_data, n_points, distances_data);
}

void containsPython(Box box, nb_array<float> points, nb_array<bool, nb::ndim<1>> contains_mask)
{
    const unsigned int n_points = points.shape(0);
    vec3<float>* points_data = (vec3<float>*) (points.data());
    bool* contains_mask_data = (bool*) (contains_mask.data());
    box.contains(points_data, n_points, contains_mask_data);
}

}; }; // end namespace freud::box

#endif
