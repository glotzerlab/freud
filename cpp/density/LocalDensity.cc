// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <complex>
#include <stdexcept>

#include "LocalDensity.h"

using namespace std;
using namespace tbb;

/*! \file LocalDensity.cc
    \brief Routines for computing local density around a point.
*/

namespace freud { namespace density {

LocalDensity::LocalDensity(float rcut, float volume, float diameter)
    : m_box(box::Box()), m_rcut(rcut), m_volume(volume), m_diameter(diameter), m_n_ref(0)
    {
    }

LocalDensity::~LocalDensity()
    {
    }

void LocalDensity::compute(const box::Box &box,
                           const freud::locality::NeighborList *nlist,
                           const vec3<float> *ref_points, unsigned int n_ref,
                           const vec3<float> *points, unsigned int Np)
    {
    m_box = box;

    nlist->validate(n_ref, Np);
    const size_t *neighbor_list(nlist->getNeighbors());

    // reallocate the output array if it is not the right size
    if (n_ref != m_n_ref)
        {
        m_density_array = std::shared_ptr<float>(new float[n_ref], std::default_delete<float[]>());
        m_num_neighbors_array = std::shared_ptr<float>(new float[n_ref], std::default_delete<float[]>());
        }

    // compute the local density
    parallel_for(blocked_range<size_t>(0, n_ref),
      [=] (const blocked_range<size_t>& r)
      {
      size_t bond(nlist->find_first_index(r.begin()));

      for(size_t i=r.begin(); i != r.end(); ++i)
          {
          float num_neighbors = 0;

          const vec3<float> r_i(ref_points[i]);

          for(; bond < nlist->getNumBonds() && neighbor_list[2*bond] == i; ++bond)
          {
              const unsigned int j(neighbor_list[2*bond + 1]);

              const vec3<float> r_j(points[j]);
              const vec3<float> r_ij(m_box.wrap(r_j - r_i));

              float rsq = dot(r_ij, r_ij);
              float r = sqrt(rsq);

              // count particles that are fully in the rcut sphere
              if (r < (m_rcut - m_diameter/2.0f))
              {
                  num_neighbors += 1.0f;
              }
              else if (r < (m_rcut + m_diameter/2.0f))
              {
                  // partially count particles that intersect the rcut sphere
                  // this is not particularly accurate for a single particle, but works well on average for
                  // lots of them. It smooths out the neighbor count distributions and avoids noisy spikes
                  // that obscure data
                  num_neighbors += 1.0f + (m_rcut - (r + m_diameter/2.0f)) / m_diameter;
              }
          }

          m_num_neighbors_array.get()[i] = num_neighbors;
          if (m_box.is2D())
              {
              // local density is area of particles divided by the area of the circle
              m_density_array.get()[i] = (m_volume * m_num_neighbors_array.get()[i]) / (M_PI * m_rcut * m_rcut);
              }
          else
              {
              // local density is volume of particles divided by the volume of the sphere
              m_density_array.get()[i] = (m_volume * m_num_neighbors_array.get()[i]) / (4.0f/3.0f * M_PI * m_rcut * m_rcut * m_rcut);
              }
          }
      });

    // save the last computed number of particles
    m_n_ref = n_ref;
    }

unsigned int LocalDensity::getNRef()
    {
    return m_n_ref;
    }

//! Get a reference to the last computed density
std::shared_ptr<float> LocalDensity::getDensity()
    {
    return m_density_array;
    }

    //! Get a reference to the last computed number of neighbors
std::shared_ptr<float> LocalDensity::getNumNeighbors()
    {
    return m_num_neighbors_array;
    }

}; }; // end namespace freud::density
