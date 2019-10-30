// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <cmath>
#include <stdexcept>
#include <tbb/tbb.h>

#include "GaussianDensity.h"

/*! \file GaussianDensity.cc
    \brief Routines for computing Gaussian smeared densities from points.
*/

namespace freud { namespace density {

GaussianDensity::GaussianDensity(vec3<unsigned int> width, float r_max, float sigma)
    : m_box(box::Box()), m_width(width), m_r_max(r_max), m_sigma(sigma)
{
    if (r_max <= 0.0f)
        throw std::invalid_argument("GaussianDensity requires r_max to be positive.");
}

//! Get a reference to the last computed Density
const util::ManagedArray<float>& GaussianDensity::getDensity() const
{
    return m_density_array;
}

//! Get width.
vec3<unsigned int> GaussianDensity::getWidth()
{
    return m_width;
}

//! internal
/*! \brief Function to compute the density array
 */
void GaussianDensity::compute(const freud::locality::NeighborQuery* nq)
{
    auto box = nq->getBox();
    auto n_points = nq->getNPoints();
    m_box = box;

    vec3<unsigned int> width(m_width);
    if (box.is2D())
    {
        width.z = 1;
    }
    m_density_array.prepare({width.x, width.y, width.z});
    util::ThreadStorage<float> local_bin_counts({width.x, width.y, width.z});

    // set up some constants first
    const float lx = m_box.getLx();
    const float ly = m_box.getLy();
    const float lz = m_box.getLz();

    const float grid_size_x = lx / m_width.x;
    const float grid_size_y = ly / m_width.y;
    const float grid_size_z = m_box.is2D() ? 0 : lz / m_width.z;

    // Find the number of bins within r_max
    const int bin_cut_x = int(m_r_max / grid_size_x);
    const int bin_cut_y = int(m_r_max / grid_size_y);
    const int bin_cut_z = m_box.is2D() ? 0 : int(m_r_max / grid_size_z);

    const float sigmasq = m_sigma * m_sigma;
    const float A = std::sqrt(1.0f / (2.0f * M_PI * sigmasq));

    util::forLoopWrapper(0, n_points, [&](size_t begin, size_t end) {
        // for each reference point
        for (size_t idx = begin; idx < end; ++idx)
        {
            const vec3<float> point = (*nq)[idx];
            // Find which bin the particle is in
            int bin_x = int((point.x + lx / 2.0f) / grid_size_x);
            int bin_y = int((point.y + ly / 2.0f) / grid_size_y);
            int bin_z = int((point.z + lz / 2.0f) / grid_size_z);

            // In 2D, only loop over the z=0 plane
            if (m_box.is2D())
            {
                bin_z = 0;
            }

            // Only evaluate over bins that are within the cutoff
            for (int k = bin_z - bin_cut_z; k <= bin_z + bin_cut_z; k++)
            {
                const float dz = float((grid_size_z * k + grid_size_z / 2.0f) - point.z - lz / 2.0f);

                for (int j = bin_y - bin_cut_y; j <= bin_y + bin_cut_y; j++)
                {
                    const float dy = float((grid_size_y * j + grid_size_y / 2.0f) - point.y - ly / 2.0f);

                    for (int i = bin_x - bin_cut_x; i <= bin_x + bin_cut_x; i++)
                    {
                        // Calculate the distance from the particle to the grid cell
                        const float dx = float((grid_size_x * i + grid_size_x / 2.0f) - point.x - lx / 2.0f);
                        vec3<float> delta = m_box.wrap(vec3<float>(dx, dy, dz));

                        const float r_sq = dot(delta, delta);
                        const float r_sqrt = std::sqrt(r_sq);

                        // Check to see if this distance is within the specified r_max
                        if (r_sqrt < m_r_max)
                        {
                            // Evaluate the gaussian ...
                            const float x_gaussian
                                = A * exp((-1.0f) * (delta.x * delta.x) / (2.0f * sigmasq));
                            const float y_gaussian
                                = A * exp((-1.0f) * (delta.y * delta.y) / (2.0f * sigmasq));
                            const float z_gaussian
                                = A * exp((-1.0f) * (delta.z * delta.z) / (2.0f * sigmasq));

                            // Assure that out of range indices are corrected for storage
                            // in the array i.e. bin -1 is actually bin 29 for nbins = 30
                            const unsigned int ni = (i + m_width.x) % m_width.x;
                            const unsigned int nj = (j + m_width.y) % m_width.y;
                            const unsigned int nk = (k + m_width.z) % m_width.z;

                            // store the product of these values in an array - n[i, j, k]
                            // = gx*gy*gz
                            local_bin_counts.local()(ni, nj, nk) += x_gaussian * y_gaussian * z_gaussian;
                        }
                    }
                }
            }
        }
    });

    // Now reduce all the arrays into one.
    util::forLoopWrapper(0, m_density_array.size(), [=](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i)
        {
            for (util::ThreadStorage<float>::const_iterator local_bins = local_bin_counts.begin();
                 local_bins != local_bin_counts.end(); ++local_bins)
            {
                m_density_array[i] += (*local_bins)[i];
            }
        }
    });
}

}; }; // end namespace freud::density
