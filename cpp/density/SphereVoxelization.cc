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
    : m_width(width), m_r_max(r_max)
{
    // dummy initial box, dimensionality of the calculation will be defined by
    // the first neighbor query passed to the compute method
    m_box = NULL;

    if (r_max <= 0.0f)
        throw std::invalid_argument("SphereVoxelization requires r_max to be positive.");
}

//! Get a reference to the last computed voxels.
const util::ManagedArray<unsigned int>& SphereVoxelization::getVoxels() const
{
    return m_voxels_array;
}

//! Get width.
vec3<unsigned int> SphereVoxelization::getWidth() const
{
    return m_width;
}

//! Compute the voxels array.
void SphereVoxelization::compute(const freud::locality::NeighborQuery* nq)
{
    // set the number of dimensions for the calculation the first time it is done
    if (m_box == NULL)
        m_box = nq->getBox();

    // Don't allow changes in the number of dimensions for the calculation
    // after its been set.
    if (nq->getBox().is2D() != m_box.is2D())
    {
        throw std::invalid_argument(
            "Each SphereVoxelization instance should be used to do calculations "
            "in a fixed number of dimensions. You have changed the number of "
            "dimensions used going from one calculation to another.");
    }

    m_box = nq->getBox();
    auto n_points = nq->getNPoints();

    // if the user gives a single number for width, but the nq box is 2D, and
    // we want a 2D calculation
    if (m_box.is2D())
    {
        m_width.z = 1;
    }
    m_voxels_array.prepare({m_width.x, m_width.y, m_width.z});

    // set up some constants first
    const float Lx = m_box.getLx();
    const float Ly = m_box.getLy();
    const float Lz = m_box.getLz();
    const vec3<bool> periodic = m_box.getPeriodic();

    const float grid_size_x = Lx / m_width.x;
    const float grid_size_y = Ly / m_width.y;
    const float grid_size_z = m_box.is2D() ? 0 : Lz / m_width.z;

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
            const int bin_x = int((point.x + Lx / 2.0f) / grid_size_x);
            const int bin_y = int((point.y + Ly / 2.0f) / grid_size_y);
            // In 2D, only loop over the z=0 plane
            const int bin_z = m_box.is2D() ? 0 : int((point.z + Lz / 2.0f) / grid_size_z);

            // Only evaluate over bins that are within the cutoff, rejecting bins
            // that are outside the box in aperiodic directions.
            for (int k = bin_z - bin_cut_z; k <= bin_z + bin_cut_z; k++)
            {
                if (!periodic.z && (k < 0 || k >= int(m_width.z)))
                {
                    continue;
                }
                const float dz = float((grid_size_z * k + grid_size_z / 2.0f) - point.z - Lz / 2.0f);

                for (int j = bin_y - bin_cut_y; j <= bin_y + bin_cut_y; j++)
                {
                    if (!periodic.y && (j < 0 || j >= int(m_width.y)))
                    {
                        continue;
                    }
                    const float dy = float((grid_size_y * j + grid_size_y / 2.0f) - point.y - Ly / 2.0f);

                    for (int i = bin_x - bin_cut_x; i <= bin_x + bin_cut_x; i++)
                    {
                        if (!periodic.x && (i < 0 || i >= int(m_width.x)))
                        {
                            continue;
                        }
                        const float dx = float((grid_size_x * i + grid_size_x / 2.0f) - point.x - Lx / 2.0f);

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
