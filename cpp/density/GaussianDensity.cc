// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <stdexcept>

#include "GaussianDensity.h"

using namespace std;
using namespace tbb;

/*! \file GaussianDensity.cc
    \brief Routines for computing Gaussian smeared densities from points.
*/

namespace freud { namespace density {


GaussianDensity::GaussianDensity(unsigned int width, float r_cut, float sigma)
    : m_box(box::Box()), m_width_x(width), m_width_y(width), m_width_z(width), m_rcut(r_cut), m_sigma(sigma)
    {
    if (width <= 0)
        throw invalid_argument("GaussianDensity requires width to be a positive integer.");
    if (r_cut <= 0.0f)
        throw invalid_argument("GaussianDensity requires r_cut to be positive.");
    }

GaussianDensity::GaussianDensity(unsigned int width_x, unsigned int width_y,
                                 unsigned int width_z, float r_cut, float sigma)
    : m_box(box::Box()), m_width_x(width_x), m_width_y(width_y), m_width_z(width_z), m_rcut(r_cut), m_sigma(sigma)
    {
    if (width_x <= 0 || width_y <= 0 || width_z <= 0)
        throw invalid_argument("GaussianDensity requires width to be a positive integer.");
    if (r_cut <= 0.0f)
        throw invalid_argument("GaussianDensity requires r_cut to be positive.");
    }

GaussianDensity::~GaussianDensity()
    {
    for (tbb::enumerable_thread_specific<float *>::iterator \
         i = m_local_bin_counts.begin();
         i != m_local_bin_counts.end(); ++i)
        {
        delete[] (*i);
        }
    }

void GaussianDensity::reduceDensity()
    {
    memset((void*)m_density_array.get(), 0, sizeof(float)*m_bi.getNumElements());
    // combine arrays
    parallel_for(blocked_range<size_t>(0,m_bi.getNumElements()),
      [=] (const blocked_range<size_t>& r)
      {
      for (size_t i = r.begin(); i != r.end(); i++)
          {
          for (tbb::enumerable_thread_specific<float *>::const_iterator \
               local_bins = m_local_bin_counts.begin();
               local_bins != m_local_bin_counts.end(); ++local_bins)
              {
              m_density_array.get()[i] += (*local_bins)[i];
              }
          }
      });
    }

//!Get a reference to the last computed Density
std::shared_ptr<float> GaussianDensity::getDensity()
    {
    if (m_reduce == true)
        {
        reduceDensity();
        }
    m_reduce = false;
    return m_density_array;
    }

//! Get x width
unsigned int GaussianDensity::getWidthX()
    {
    return m_width_x;
    }

//! Get y width
unsigned int GaussianDensity::getWidthY()
    {
    return m_width_y;
    }

//! Get z width
unsigned int GaussianDensity::getWidthZ()
    {
    if (!m_box.is2D())
        {
        return m_width_z;
        }
    else
        {
        return 0;
        }
    }

//! \internal
/*! \brief Function to reset the density array if needed e.g. calculating between new particle types
*/
void GaussianDensity::reset()
    {
    for (tbb::enumerable_thread_specific<float *>::iterator \
         i = m_local_bin_counts.begin();
         i != m_local_bin_counts.end(); ++i)
        {
        memset((void*)(*i), 0, sizeof(float)*m_bi.getNumElements());
        }
    this->m_reduce = true;
    }

//! internal
/*! \brief Function to compute the density array
*/
void GaussianDensity::compute(const box::Box &box, const vec3<float> *points, unsigned int Np)
    {
    reset();
    m_box = box;
    if (m_box.is2D())
        {
        m_bi = Index3D(m_width_x, m_width_y, 1);
        }
    else
        {
        m_bi = Index3D(m_width_x, m_width_y, m_width_z);
        }

    // this does not agree with rest of freud
    m_density_array = std::shared_ptr<float>(
            new float[m_bi.getNumElements()],
            std::default_delete<float[]>());
    parallel_for(blocked_range<size_t>(0,Np),
            [=] (const blocked_range<size_t>& r)
        {
        assert(points);
        assert(Np > 0);

        bool exists;
        m_local_bin_counts.local(exists);
        if (! exists)
            {
            m_local_bin_counts.local() = new float [m_bi.getNumElements()];
            memset((void*)m_local_bin_counts.local(), 0,
                    sizeof(float)*m_bi.getNumElements());
            }

        // set up some constants first
        float lx = m_box.getLx();
        float ly = m_box.getLy();
        float lz = m_box.getLz();

        float grid_size_x = lx/m_width_x;
        float grid_size_y = ly/m_width_y;
        float grid_size_z = lz/m_width_z;

        float sigmasq = m_sigma*m_sigma;
        float A = sqrt(1.0f/(2.0f*M_PI*sigmasq));

        // for each reference point
        for (size_t idx = r.begin(); idx != r.end(); idx++)
            {
            // find the distance of that particle to bins
            // will use this information to evaluate the Gaussian
            // Find the which bin the particle is in
            int bin_x = int((points[idx].x+lx/2.0f)/grid_size_x);
            int bin_y = int((points[idx].y+ly/2.0f)/grid_size_y);
            int bin_z = int((points[idx].z+lz/2.0f)/grid_size_z);

            // Find the number of bins within r_cut
            int bin_cut_x = int(m_rcut/grid_size_x);
            int bin_cut_y = int(m_rcut/grid_size_y);
            int bin_cut_z = int(m_rcut/grid_size_z);

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
                float dz = float((grid_size_z*k + grid_size_z/2.0f) - \
                        points[idx].z - lz/2.0f);

                for (int j = bin_y - bin_cut_y; j <= bin_y + bin_cut_y; j++)
                    {
                    float dy = float((grid_size_y*j + grid_size_y/2.0f) - \
                            points[idx].y - ly/2.0f);

                    for (int i = bin_x - bin_cut_x; i<= bin_x + bin_cut_x; i++)
                        {
                        // Calculate the distance from the grid cell to particular particle
                        float dx = float((grid_size_x*i + grid_size_x/2.0f) - \
                                points[idx].x - lx/2.0f);
                        vec3<float> delta = m_box.wrap(vec3<float>(dx, dy, dz));

                        float rsq = dot(delta, delta);
                        float rsqrt = sqrtf(rsq);

                        // Check to see if this distance is within the specified r_cut
                        if (rsqrt < m_rcut)
                            {
                            // Evaluate the gaussian ...
                            float x_gaussian = A*exp((-1.0f)*(delta.x*delta.x)/(2.0f*sigmasq));
                            float y_gaussian = A*exp((-1.0f)*(delta.y*delta.y)/(2.0f*sigmasq));
                            float z_gaussian = A*exp((-1.0f)*(delta.z*delta.z)/(2.0f*sigmasq));

                            // Assure that out of range indices are corrected for storage in the array
                            // i.e. bin -1 is actually bin 29 for nbins = 30
                            unsigned int ni = (i + m_width_x) % m_width_x;
                            unsigned int nj = (j + m_width_y) % m_width_y;
                            unsigned int nk = (k + m_width_z) % m_width_z;

                            // store the product of these values in an array - n[i, j, k] = gx*gy*gz
                            m_local_bin_counts.local()[m_bi(ni, nj, nk)] += \
                                x_gaussian*y_gaussian*z_gaussian;
                            }
                        }
                    }
                }
            }
        });
    }
}; }; // end namespace freud::density
