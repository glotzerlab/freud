// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <cmath>
#include <stdexcept>

#include "SphereVoxelization.h"

/*! \file SphereVoxelization.cc
    \brief Routines for computing voxelized densities from spheres centered at points.
*/

namespace freud { namespace density {

SphereVoxelization::SphereVoxelization(vec3<unsigned int> width, float r_max)
    : m_box(box::Box()), m_width(width), m_r_max(r_max)
{
    if (r_max <= 0.0f)
        throw std::invalid_argument("Sphere Voxelization requires r_max to be positive.");
}

//! Get a reference to the last computed voxels
const util::ManagedArray<unsigned int>& SphereVoxelization::getVoxels() const
{
    return m_voxels_array;
}

//! Get width.
vec3<unsigned int> SphereVoxelization::getWidth()
{
    return m_width;
}

//! internal
/*! \brief Function to compute the voxels array
 */
void SphereVoxelization::compute(const freud::locality::NeighborQuery* nq)
{
    m_box = nq->getBox();
    auto n_points = nq->getNPoints();

    vec3<unsigned int> width(m_width);
    if (m_box.is2D())
    {
        width.z = 1;
    }
    m_voxels_array.prepare({width.x, width.y, width.z});

    // set up some constants first
    const float lx = m_box.getLx();
    const float ly = m_box.getLy();
    const float lz = m_box.getLz();
    const vec3<bool> periodic = m_box.getPeriodic();

    const float grid_size_x = lx / m_width.x;
    const float grid_size_y = ly / m_width.y;
    const float grid_size_z = m_box.is2D() ? 0 : lz / m_width.z;

    // Find the number of bins within r_max
    const int bin_cut_x = int(m_r_max / grid_size_x);
    const int bin_cut_y = int(m_r_max / grid_size_y);
    const int bin_cut_z = m_box.is2D() ? 0 : int(m_r_max / grid_size_z);
    const float r_max_sq = m_r_max * m_r_max;

    util::forLoopWrapper(0, n_points, [&](size_t begin, size_t end) {
        // for each reference point
        for (size_t idx = begin; idx < end; ++idx)
        {
            const vec3<float> point = (*nq)[idx];
            // Find which bin the particle is in
            const int bin_x = int((point.x + lx / 2.0f) / grid_size_x);
            const int bin_y = int((point.y + ly / 2.0f) / grid_size_y);
            // In 2D, only loop over the z=0 plane
            const int bin_z = m_box.is2D() ? 0 : int((point.z + lz / 2.0f) / grid_size_z);

            // Reject bins that are outside the box in aperiodic directions
            // Only evaluate over bins that are within the cutoff
            for (int k = bin_z - bin_cut_z; k <= bin_z + bin_cut_z; k++)
            {
                if (!periodic.z && (k < 0 || k >= int(m_width.z)))
                {
                    continue;
                }
                const float dz = float((grid_size_z * k + grid_size_z / 2.0f) - point.z - lz / 2.0f);

                for (int j = bin_y - bin_cut_y; j <= bin_y + bin_cut_y; j++)
                {
                    if (!periodic.y && (j < 0 || j >= int(m_width.y)))
                    {
                        continue;
                    }
                    const float dy = float((grid_size_y * j + grid_size_y / 2.0f) - point.y - ly / 2.0f);

                    for (int i = bin_x - bin_cut_x; i <= bin_x + bin_cut_x; i++)
                    {
                        if (!periodic.x && (i < 0 || i >= int(m_width.x)))
                        {
                            continue;
                        }
                        const float dx = float((grid_size_x * i + grid_size_x / 2.0f) - point.x - lx / 2.0f);

                        // Calculate the distance from the particle to the grid cell
                        const vec3<float> delta = m_box.wrap(vec3<float>(dx, dy, dz));

                        const float r_sq = dot(delta, delta);

                        // Check to see if this distance is within the specified r_max
                        if (r_sq < r_max_sq)
                        {
                            // Assure that out of range indices are corrected for storage
                            // in the array i.e. bin -1 is actually bin 29 for nbins = 30
                            const unsigned int ni = (i + m_width.x) % m_width.x;
                            const unsigned int nj = (j + m_width.y) % m_width.y;
                            const unsigned int nk = (k + m_width.z) % m_width.z;

                            // This array value could be written by multiple threads in parallel.
                            // This is only safe because all threads are writing the same value (1).
                            m_voxels_array(ni, nj, nk) = 1;
                        }
                    }
                }
            }
        }
    });
}

}; }; // end namespace freud::density
