// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <cmath>
#include <stdexcept>

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

//! Compute the density array.
void GaussianDensity::compute(const freud::locality::NeighborQuery* nq)
{
    // Don't allow box dimensions to change afer a calculation has
    // already been done (m_width is altered)
    if (nq->getBox().is2D() != m_box.is2D())
    {
        throw std::invalid_argument(
            "GaussianDensity cannot compute on 3D boxes after computing 2D boxes.");
    }

    m_box = nq->getBox();
    auto n_points = nq->getNPoints();

    if (m_box.is2D())
    {
        m_width.z = 1;
    }
    m_density_array.prepare({m_width.x, m_width.y, m_width.z});
    util::ThreadStorage<float> local_bin_counts({m_width.x, m_width.y, m_width.z});

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
    const float sigmasq = m_sigma * m_sigma;
    const float normalization_base = 1.0f / std::sqrt(constants::TWO_PI * sigmasq);
    const float dimensions = m_box.is2D() ? 2.0f : 3.0f;
    const float normalization = std::pow(normalization_base, dimensions);

    util::forLoopWrapper(0, n_points, [&](size_t begin, size_t end) {
        // for each reference point
        for (size_t idx = begin; idx < end; ++idx)
        {
            const vec3<float> point = (*nq)[idx];
            // Find which bin the particle is in
            int bin_x = int((point.x + Lx / 2.0f) / grid_size_x);
            int bin_y = int((point.y + Ly / 2.0f) / grid_size_y);
            int bin_z = int((point.z + Lz / 2.0f) / grid_size_z);

            // In 2D, only loop over the z=0 plane
            if (m_box.is2D())
            {
                bin_z = 0;
            }

            // Reject bins that are outside the box in aperiodic directions
            // Only evaluate over bins that are within the cutoff
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
                            // Evaluate the gaussian
                            const float gaussian = normalization * std::exp(-r_sq / (2 * sigmasq));

                            // Assure that out of range indices are corrected for storage
                            // in the array i.e. bin -1 is actually bin 29 for nbins = 30
                            const unsigned int ni = (i + m_width.x) % m_width.x;
                            const unsigned int nj = (j + m_width.y) % m_width.y;
                            const unsigned int nk = (k + m_width.z) % m_width.z;

                            // Store the gaussian contribution
                            local_bin_counts.local()(ni, nj, nk) += gaussian;
                        }
                    }
                }
            }
        }
    });

    // Parallel reduction over thread storage
    local_bin_counts.reduceInto(m_density_array);
}

}; }; // end namespace freud::density
