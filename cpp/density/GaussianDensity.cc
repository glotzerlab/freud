// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

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
const util::ManagedArray<float> &GaussianDensity::getDensity() const
{
    return m_density_array;
}

//! Get width.
vec3<unsigned int> GaussianDensity::getWidth()
{
    return m_width;
}

//! \internal
/*! \brief Function to reset the density array if needed e.g. calculating between new particle types
 */
void GaussianDensity::reset()
{
    m_local_bin_counts.reset();
}

//! internal
/*! \brief Function to compute the density array
 */
void GaussianDensity::compute(const box::Box& box, const vec3<float>* points, unsigned int n_points)
{
    reset();
    m_box = box;

    vec3<unsigned int> width(m_width);
    if (box.is2D())
    {
        width.z = 1;
    }
    m_density_array.prepare({width.x, width.y, width.z});
    m_local_bin_counts.resize({width.x, width.y, width.z});
    util::forLoopWrapper(0, n_points, [=](size_t begin, size_t end) {
        // set up some constants first
        float lx = m_box.getLx();
        float ly = m_box.getLy();
        float lz = m_box.getLz();

        float grid_size_x = lx / m_width.x;
        float grid_size_y = ly / m_width.y;
        float grid_size_z = lz / m_width.z;

        float sigmasq = m_sigma * m_sigma;
        float A = sqrt(1.0f / (2.0f * M_PI * sigmasq));

        // for each reference point
        for (size_t idx = begin; idx < end; ++idx)
        {
            // find the distance of that particle to bins
            // will use this information to evaluate the Gaussian
            // Find the which bin the particle is in
            int bin_x = int((points[idx].x + lx / 2.0f) / grid_size_x);
            int bin_y = int((points[idx].y + ly / 2.0f) / grid_size_y);
            int bin_z = int((points[idx].z + lz / 2.0f) / grid_size_z);

            // Find the number of bins within r_max
            int bin_cut_x = int(m_r_max / grid_size_x);
            int bin_cut_y = int(m_r_max / grid_size_y);
            int bin_cut_z = int(m_r_max / grid_size_z);

            // in 2D, only loop over the 0 z plane
            if (m_box.is2D())
            {
                bin_z = 0;
                bin_cut_z = 0;
                grid_size_z = 0;
            }
            // Only evaluate over bins that are within the cut off
            // to reduce the number of computations
            for (int k = bin_z - bin_cut_z; k <= bin_z + bin_cut_z; k++)
            {
                float dz = float((grid_size_z * k + grid_size_z / 2.0f) - points[idx].z - lz / 2.0f);

                for (int j = bin_y - bin_cut_y; j <= bin_y + bin_cut_y; j++)
                {
                    float dy = float((grid_size_y * j + grid_size_y / 2.0f) - points[idx].y - ly / 2.0f);

                    for (int i = bin_x - bin_cut_x; i <= bin_x + bin_cut_x; i++)
                    {
                        // Calculate the distance from the grid cell to particular particle
                        float dx = float((grid_size_x * i + grid_size_x / 2.0f) - points[idx].x - lx / 2.0f);
                        vec3<float> delta = m_box.wrap(vec3<float>(dx, dy, dz));

                        float r_sq = dot(delta, delta);
                        float r_sqrt = sqrtf(r_sq);

                        // Check to see if this distance is within the specified r_max
                        if (r_sqrt < m_r_max)
                        {
                            // Evaluate the gaussian ...
                            float x_gaussian = A * exp((-1.0f) * (delta.x * delta.x) / (2.0f * sigmasq));
                            float y_gaussian = A * exp((-1.0f) * (delta.y * delta.y) / (2.0f * sigmasq));
                            float z_gaussian = A * exp((-1.0f) * (delta.z * delta.z) / (2.0f * sigmasq));

                            // Assure that out of range indices are corrected for storage
                            // in the array i.e. bin -1 is actually bin 29 for nbins = 30
                            unsigned int ni = (i + m_width.x) % m_width.x;
                            unsigned int nj = (j + m_width.y) % m_width.y;
                            unsigned int nk = (k + m_width.z) % m_width.z;

                            // store the product of these values in an array - n[i, j, k]
                            // = gx*gy*gz
                            m_local_bin_counts.local()(ni, nj, nk) += x_gaussian * y_gaussian * z_gaussian;
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
            for (util::ThreadStorage<float>::const_iterator local_bins = m_local_bin_counts.begin();
                 local_bins != m_local_bin_counts.end(); ++local_bins)
            {
                m_density_array[i] += (*local_bins)[i];
            }
        }
    });
}

}; }; // end namespace freud::density
