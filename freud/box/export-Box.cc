// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <stdexcept>

#include "Box.h"
#include "export-Box.h"
#include "VectorMath.h"

namespace nb = nanobind;

namespace freud { namespace box { namespace wrap {

void makeAbsolute(const std::shared_ptr<Box>& box, nb_array<float> vecs, nb_array<float> out)
{
    unsigned int Nvecs = vecs.shape(0);
    auto* vecs_data = reinterpret_cast<vec3<float>*>(vecs.data());
    auto* out_data = reinterpret_cast<vec3<float>*>(out.data());
    box->makeAbsolute(vecs_data, Nvecs, out_data);
}

void makeFractional(const std::shared_ptr<Box>& box, nb_array<float> vecs, nb_array<float> out)
{
    unsigned int Nvecs = vecs.shape(0);
    auto* vecs_data = reinterpret_cast<vec3<float>*>(vecs.data());
    auto* out_data = reinterpret_cast<vec3<float>*>(out.data());
    box->makeFractional(vecs_data, Nvecs, out_data);
}

void getImages(const std::shared_ptr<Box>& box, nb_array<float> vecs, nb_array<int> images)
{
    const unsigned int Nvecs = vecs.shape(0);
    auto* vecs_data = reinterpret_cast<vec3<float>*>(vecs.data());
    auto* images_data = reinterpret_cast<vec3<int>*>(images.data());
    box->getImages(vecs_data, Nvecs, images_data);
}

void wrap(const std::shared_ptr<Box>& box, nb_array<float> vecs, nb_array<float> out)
{
    const unsigned int Nvecs = vecs.shape(0);
    auto* vecs_data = reinterpret_cast<vec3<float>*>(vecs.data());
    auto* out_data = reinterpret_cast<vec3<float>*>(out.data());
    box->wrap(vecs_data, Nvecs, out_data);
}

void unwrap(const std::shared_ptr<Box>& box, nb_array<float> vecs, nb_array<int> images, nb_array<float> out)
{
    const unsigned int Nvecs = vecs.shape(0);
    auto* vecs_data = reinterpret_cast<vec3<float>*>(vecs.data());
    auto* images_data = reinterpret_cast<vec3<int>*>(images.data());
    auto* out_data = reinterpret_cast<vec3<float>*>(out.data());
    box->unwrap(vecs_data, images_data, Nvecs, out_data);
}

std::vector<float> centerOfMass(const std::shared_ptr<Box>& box, nb_array<float> vecs,
                                nb_array<float, nb::shape<-1>> masses)
{
    const unsigned int Nvecs = vecs.shape(0);
    auto* vecs_data = reinterpret_cast<vec3<float>*>(vecs.data());
    auto* masses_data = reinterpret_cast<float*>(masses.data());
    auto com = box->centerOfMass(vecs_data, Nvecs, masses_data);
    return {com.x, com.y, com.z};
}

void center(const std::shared_ptr<Box>& box, nb_array<float> vecs, nb_array<float, nb::ndim<1>> masses)
{
    const unsigned int Nvecs = vecs.shape(0);
    auto* vecs_data = reinterpret_cast<vec3<float>*>(vecs.data());
    auto* masses_data = reinterpret_cast<float*>(masses.data());
    box->center(vecs_data, Nvecs, masses_data);
}

void computeDistances(const std::shared_ptr<Box>& box, nb_array<float> query_points, nb_array<float> points,
                      nb_array<float, nb::ndim<1>> distances)
{
    const unsigned int n_query_points = query_points.shape(0);
    auto* query_points_data = reinterpret_cast<vec3<float>*>(query_points.data());
    const unsigned int n_points = points.shape(0);
    auto* points_data = reinterpret_cast<vec3<float>*>(points.data());
    auto* distances_data = reinterpret_cast<float*>(distances.data());
    if (n_query_points != n_points)
    {
        throw std::invalid_argument("The number of query points and points must match.");
    }
    box->computeDistances(query_points_data, n_query_points, points_data, distances_data);
}

void computeAllDistances(const std::shared_ptr<Box>& box, nb_array<float> query_points,
                         nb_array<float> points, nb_array<float, nb::ndim<2>> distances)
{
    const unsigned int n_query_points = query_points.shape(0);
    auto* query_points_data = reinterpret_cast<vec3<float>*>(query_points.data());
    const unsigned int n_points = points.shape(0);
    auto* points_data = reinterpret_cast<vec3<float>*>(points.data());
    auto* distances_data = reinterpret_cast<float*>(distances.data());
    box->computeAllDistances(query_points_data, n_query_points, points_data, n_points, distances_data);
}

void contains(const std::shared_ptr<Box>& box, nb_array<float> points,
              nb_array<bool, nb::ndim<1>> contains_mask)
{
    const unsigned int n_points = points.shape(0);
    auto* points_data = reinterpret_cast<vec3<float>*>(points.data());
    auto* contains_mask_data = reinterpret_cast<bool*>(contains_mask.data());
    box->contains(points_data, n_points, contains_mask_data);
}

}; }; }; // namespace freud::box::wrap
