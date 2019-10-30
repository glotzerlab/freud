// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include "PMFTXYZ.h"
#include <stdexcept>

/*! \file PMFTXYZ.cc
    \brief Routines for computing 3D potential of mean force in XYZ coordinates
*/

namespace freud { namespace pmft {

PMFTXYZ::PMFTXYZ(float x_max, float y_max, float z_max, unsigned int n_x, unsigned int n_y, unsigned int n_z,
                 vec3<float> shiftvec)
    : PMFT(), m_shiftvec(shiftvec)
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

    // Compute Jacobian
    const float dx = float(2.0) * x_max / float(n_x);
    const float dy = float(2.0) * y_max / float(n_y);
    const float dz = float(2.0) * z_max / float(n_z);

    // To properly normalize, we must consider the 3 degrees of freedom that
    // are implicitly integrated over by this calculation. In three dimensions,
    // there are 6 total degrees of freedom; we are only accounting for the
    // relative positions. The orientational degrees of freedom of the second
    // particle constitute the other three. The total volume in our 6D
    // coordinate space accounted for by these orientational degrees of freedom
    // is 8*pi^2. To see this, consider an integral over the Euler angles as shown
    // here https://en.wikipedia.org/wiki/3D_rotation_group#Spherical_harmonics
    // or discussed more in depth in Representations of the Rotation and
    // Lorentz Groups and Their Applications by Gelfand, Minlos, and Shapiro.
    const float orientation_volume = 8 * M_PI * M_PI;
    m_jacobian = dx * dy * dz * orientation_volume;

    // create and populate the pcf_array
    m_pcf_array.prepare({n_x, n_y, n_z});

    // Construct the Histogram object that will be used to keep track of counts of bond distances found.
    BHAxes axes;
    axes.push_back(std::make_shared<util::RegularAxis>(n_x, -x_max, x_max));
    axes.push_back(std::make_shared<util::RegularAxis>(n_y, -y_max, y_max));
    axes.push_back(std::make_shared<util::RegularAxis>(n_z, -z_max, z_max));
    m_histogram = BondHistogram(axes);
    m_local_histograms = BondHistogram::ThreadLocalHistogram(m_histogram);
}

//! \internal
//! helper function to reduce the thread specific arrays into one array
void PMFTXYZ::reduce()
{
    float jacobian_factor = (float) 1.0 / m_jacobian;
    PMFT::reduce([jacobian_factor](size_t i) { return jacobian_factor; });
}

//! \internal
/*! \brief Helper function to direct the calculation to the correct helper class
 */
void PMFTXYZ::accumulate(const locality::NeighborQuery* neighbor_query, quat<float>* query_orientations,
                         vec3<float>* query_points, unsigned int n_query_points,
                         quat<float>* face_orientations, unsigned int n_faces,
                         const locality::NeighborList* nlist, freud::locality::QueryArgs qargs)
{
    // precalc some values for faster computation within the loop
    neighbor_query->getBox().enforce3D();
    accumulateGeneral(neighbor_query, query_points, n_query_points, nlist, qargs,
                      [=](const freud::locality::NeighborBond& neighbor_bond) {
                          // create the reference point quaternion
                          quat<float> ref_q(query_orientations[neighbor_bond.point_idx]);
                          // make sure that the particles are wrapped into the box
                          vec3<float> delta(bondVector(neighbor_bond, neighbor_query, query_points));

                          for (unsigned int k = 0; k < n_faces; k++)
                          {
                              // create the extra quaternion
                              quat<float> qe(face_orientations[util::ManagedArray<unsigned int>::getIndex(
                                  {neighbor_query->getNPoints(), n_faces}, {neighbor_bond.point_idx, k})]);
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
