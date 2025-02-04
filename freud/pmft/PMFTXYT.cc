// Copyright (c) 2010-2025 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Box.h"
#include "Histogram.h"
#include "ManagedArray.h"
#include "NeighborBond.h"
#include "NeighborList.h"
#include "NeighborQuery.h"
#include "PMFT.h"
#include "PMFTXYT.h"
#include "VectorMath.h"
#include "utils.h"

/*! \file PMFTXYT.cc
    \brief Routines for computing potential of mean force and torque in XYT coordinates
*/

namespace freud { namespace pmft {

PMFTXYT::PMFTXYT(float x_max, float y_max, unsigned int n_x, unsigned int n_y, unsigned int n_t) : PMFT()
{
    if (n_x < 1)
    {
        throw std::invalid_argument("PMFTXYT requires at least 1 bin in X.");
    }
    if (n_y < 1)
    {
        throw std::invalid_argument("PMFTXYT requires at least 1 bin in Y.");
    }
    if (n_t < 1)
    {
        throw std::invalid_argument("PMFTXYT requires at least 1 bin in T.");
    }
    if (x_max < 0)
    {
        throw std::invalid_argument("PMFTXYT requires that x_max must be positive.");
    }
    if (y_max < 0)
    {
        throw std::invalid_argument("PMFTXYT requires that y_max must be positive.");
    }

    // Note: There is an additional implicit volume factor of 2*pi
    // corresponding to the rotational degree of freedom in the system (i.e. dt
    // technically has 2*pi in the numerator). However, this factor is
    // implicitly canceled out since we also do not include it in the number
    // density computed for the system, see PMFT::reduce for more information.
    const float dx = float(2.0) * x_max / float(n_x);
    const float dy = float(2.0) * y_max / float(n_y);
    const float dt = 1 / float(n_t);
    m_jacobian = dx * dy * dt;

    // Create the PCF array.
    m_pcf_array = std::make_shared<util::ManagedArray<float>>(std::vector<size_t> {n_x, n_y, n_t});

    // Construct the Histogram object that will be used to keep track of counts of bond distances found.
    const auto axes = util::Axes {std::make_shared<util::RegularAxis>(n_x, -x_max, x_max),
                                  std::make_shared<util::RegularAxis>(n_y, -y_max, y_max),
                                  std::make_shared<util::RegularAxis>(n_t, 0, constants::TWO_PI)};
    m_histogram = BondHistogram(axes);
    m_local_histograms = BondHistogram::ThreadLocalHistogram(m_histogram);
}

void PMFTXYT::reduce()
{
    float const jacobian_factor = (float) 1.0 / m_jacobian;
    PMFT::reduce([jacobian_factor](size_t i) { return jacobian_factor; }); // NOLINT(misc-unused-parameters)
}

void PMFTXYT::accumulate(const std::shared_ptr<locality::NeighborQuery>& neighbor_query,
                         const float* orientations, const vec3<float>* query_points,
                         const float* query_orientations, unsigned int n_query_points,
                         std::shared_ptr<locality::NeighborList> nlist,
                         const freud::locality::QueryArgs& qargs)
{
    neighbor_query->getBox().enforce2D();
    accumulateGeneral(neighbor_query, query_points, n_query_points, std::move(nlist), qargs,
                      [&](const freud::locality::NeighborBond& neighbor_bond) {
                          const vec3<float>& delta(neighbor_bond.getVector());

                          // rotate interparticle vector
                          const vec2<float> myVec(delta.x, delta.y);
                          const rotmat2<float> myMat(rotmat2<float>::fromAngle(
                              -query_orientations[neighbor_bond.getQueryPointIdx()]));
                          const vec2<float> rotVec = myMat * myVec;
                          // calculate angle
                          const float d_theta = std::atan2(-delta.y, -delta.x);
                          // make sure that t is bounded between 0 and 2PI
                          const float t = util::modulusPositive(
                              orientations[neighbor_bond.getPointIdx()] - d_theta, constants::TWO_PI);
                          m_local_histograms(rotVec.x, rotVec.y, t);
                      });
}
}; }; // end namespace freud::pmft
