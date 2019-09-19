// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <stdexcept>
#include "PMFTXYZ.h"

/*! \file PMFTXYZ.cc
    \brief Routines for computing 3D potential of mean force in XYZ coordinates
*/

namespace freud { namespace pmft {

PMFTXYZ::PMFTXYZ(float x_max, float y_max, float z_max, unsigned int n_x, unsigned int n_y, unsigned int n_z,
                 vec3<float> shiftvec)
    : PMFT(),
      m_shiftvec(shiftvec)
{
    if (n_x < 1)
        throw std::invalid_argument("PMFTXYZ requires at least 1 bin in X.");
    if (n_y < 1)
        throw std::invalid_argument("PMFTXYZ requires at least 1 bin in Y.");
    if (n_z < 1)
        throw std::invalid_argument("PMFTXYZ requires at least 1 bin in Z.");
    if (x_max < 0.0f)
        throw std::invalid_argument("PMFTXYZ requires that x_max must be positive.");
    if (y_max < 0.0f)
        throw std::invalid_argument("PMFTXYZ requires that y_max must be positive.");
    if (z_max < 0.0f)
        throw std::invalid_argument("PMFTXYZ requires that z_max must be positive.");

    // calculate dx, dy, dz
    float dx = float(2.0) * x_max / float(n_x);
    float dy = float(2.0) * y_max / float(n_y);
    float dz = float(2.0) * z_max / float(n_z);
    m_jacobian = dx * dy * dz;

    // create and populate the pcf_array
    m_pcf_array.prepare({n_x, n_y, n_z});

    // Construct the Histogram object that will be used to keep track of counts of bond distances found.
    util::Histogram::Axes axes;
    axes.push_back(std::make_shared<util::RegularAxis>(n_x, -x_max, x_max));
    axes.push_back(std::make_shared<util::RegularAxis>(n_y, -y_max, y_max));
    axes.push_back(std::make_shared<util::RegularAxis>(n_z, -z_max, z_max));
    m_histogram = util::Histogram(axes);
    m_local_histograms = util::Histogram::ThreadLocalHistogram(m_histogram);
}

//! \internal
//! helper function to reduce the thread specific arrays into one array
void PMFTXYZ::reducePCF()
{
    float jacobian_factor = (float) 1.0 / m_jacobian;
    reduce([jacobian_factor](size_t i) { return jacobian_factor; });
}

//! \internal
/*! \brief Helper function to direct the calculation to the correct helper class
 */
void PMFTXYZ::accumulate(const locality::NeighborQuery* neighbor_query,
                         quat<float>* orientations, vec3<float>* query_points,
                         unsigned int n_query_points, quat<float>* face_orientations,
                         unsigned int n_faces, const locality::NeighborList* nlist,
                         freud::locality::QueryArgs qargs)
{
    // precalc some values for faster computation within the loop
    Index2D q_i = Index2D(n_faces, n_query_points);

    std::vector<unsigned int> shape = m_local_histograms.local().shape();
    accumulateGeneral(neighbor_query, query_points, n_query_points, nlist, qargs,
        [=](const freud::locality::NeighborBond& neighbor_bond) {
        vec3<float> ref = neighbor_query->getPoints()[neighbor_bond.ref_id];
        // create the reference point quaternion
        quat<float> ref_q(orientations[neighbor_bond.ref_id]);
        // make sure that the particles are wrapped into the box
        vec3<float> delta = m_box.wrap(query_points[neighbor_bond.id] - ref);

        for (unsigned int k = 0; k < n_faces; k++)
        {
            // create the extra quaternion
            quat<float> qe(face_orientations[q_i(k, neighbor_bond.ref_id)]);
            // create point vector
            vec3<float> v(delta);
            // rotate the vector
            v = rotate(conj(ref_q), v);
            v = rotate(qe, v);

            m_local_histograms(v.x, v.y, v.z);
        }
    });
}

}; }; // end namespace freud::pmft
