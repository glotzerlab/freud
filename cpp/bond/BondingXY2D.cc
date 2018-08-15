// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <complex>
#include <map>
#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include "BondingXY2D.h"

using namespace std;
using namespace tbb;

namespace freud { namespace bond {

BondingXY2D::BondingXY2D(float x_max,
                         float y_max,
                         unsigned int n_bins_x,
                         unsigned int n_bins_y,
                         unsigned int n_bonds,
                         unsigned int *bond_map,
                         unsigned int *bond_list)
    : m_box(box::Box()), m_x_max(x_max), m_y_max(y_max), m_nbins_x(n_bins_x), m_nbins_y(n_bins_y),
      m_n_bonds(n_bonds), m_bond_map(bond_map), m_bond_list(bond_list), m_n_ref(0), m_n_p(0)
    {
    // create the unsigned int array to store whether or not a particle is paired
    m_bonds = std::shared_ptr<unsigned int>(new unsigned int[m_n_ref*m_n_bonds], std::default_delete<unsigned int[]>());
    if (n_bins_x < 1)
        throw invalid_argument("must be at least 1 bin in x");
    if (n_bins_y < 1)
        throw invalid_argument("must be at least 1 bin in y");
    if (x_max < 0.0f)
        throw invalid_argument("x_max must be positive");
    if (y_max < 0.0f)
        throw invalid_argument("y_max must be positive");
    // calculate dx, dy
    m_dx = 2.0 * m_x_max / float(m_nbins_x);
    m_dy = 2.0 * m_y_max / float(m_nbins_y);

    if (m_dx > x_max)
        throw invalid_argument("x_max must be greater than dx");
    if (m_dy > y_max)
        throw invalid_argument("y_max must be greater than dy");

    // create mapping between bond index and list index
    for (unsigned int i = 0; i < m_n_bonds; i++)
        {
        m_list_map[m_bond_list[i]] = i;
        }

    // create mapping between list index and bond index
    for (unsigned int i = 0; i < m_n_bonds; i++)
        {
        m_rev_list_map[i] = m_bond_list[i];
        }

    m_r_max = sqrtf(m_x_max*m_x_max + m_y_max*m_y_max);
    }

BondingXY2D::~BondingXY2D()
    {
    }

std::shared_ptr<unsigned int> BondingXY2D::getBonds()
    {
    return m_bonds;
    }

std::map<unsigned int, unsigned int> BondingXY2D::getListMap()
    {
    return m_list_map;
    }

std::map<unsigned int, unsigned int> BondingXY2D::getRevListMap()
    {
    return m_rev_list_map;
    }

void BondingXY2D::compute(box::Box& box,
                          const freud::locality::NeighborList *nlist,
                          vec3<float> *ref_points,
                          float *ref_orientations,
                          unsigned int n_ref,
                          vec3<float> *points,
                          float *orientations,
                          unsigned int n_p)
    {
    m_box = box;

    nlist->validate(n_ref, n_p);
    const size_t *neighbor_list(nlist->getNeighbors());

    if (n_ref != m_n_ref)
        {
        // make sure to clear this out at some point
        m_bonds = std::shared_ptr<unsigned int>(new unsigned int[n_ref*m_n_bonds], std::default_delete<unsigned int[]>());
        }

    std::fill(m_bonds.get(), m_bonds.get()+int(n_ref*m_n_bonds), UINT_MAX);
    // compute the order parameter
    parallel_for(blocked_range<size_t>(0,n_ref),
        [=] (const blocked_range<size_t>& br)
            {
            float dx_inv = 1.0f / m_dx;
            float dy_inv = 1.0f / m_dy;
            // indexer for bond list
            Index2D a_i = Index2D(m_n_bonds, n_ref);
            // indexer for bond map
            Index2D b_i = Index2D(m_nbins_x, m_nbins_y);
            size_t bond(nlist->find_first_index(br.begin()));

            for(size_t i=br.begin(); i!=br.end(); ++i)
                {
                // get position, orientation of particle i
                vec3<float> ref_pos = ref_points[i];

                for(; bond < nlist->getNumBonds() && neighbor_list[2*bond] == i; ++bond)
                    {
                    const size_t j(neighbor_list[2*bond + 1]);
                        {
                        //compute r between the two particles
                        vec3<float> delta = m_box.wrap(points[j] - ref_pos);

                        float rsq = dot(delta, delta);
                        // particle cannot pair with itself...i != j is probably better?
                        if (rsq < 1e-6)
                            {
                            continue;
                            }
                        // if particle is not outside of possible radius
                        if (rsq < m_r_max)
                            {
                            /// rotate interparticle vector
                            vec2<float> myVec(delta.x, delta.y);
                            rotmat2<float> myMat = rotmat2<float>::fromAngle(-ref_orientations[i]);
                            vec2<float> rotVec = myMat * myVec;
                            float x = rotVec.x + m_x_max;
                            float y = rotVec.y + m_y_max;

                            // find the bin to increment
                            float binx = floorf(x * dx_inv);
                            float biny = floorf(y * dy_inv);
                            // fast float to int conversion with truncation
                            #ifdef __SSE2__
                            unsigned int ibinx = _mm_cvtt_ss2si(_mm_load_ss(&binx));
                            unsigned int ibiny = _mm_cvtt_ss2si(_mm_load_ss(&biny));
                            #else
                            unsigned int ibinx = (unsigned int)(binx);
                            unsigned int ibiny = (unsigned int)(biny);
                            #endif

                            // log the bond
                            if ((ibinx < m_nbins_x) && (ibiny < m_nbins_y))
                                {
                                // find the bond that corresponds to this point
                                unsigned int bond = m_bond_map[b_i(ibinx, ibiny)];
                                // get the index from the map
                                auto list_idx = m_list_map.find(bond);
                                // bin if bond is tracked
                                if (list_idx != m_list_map.end())
                                    {
                                    m_bonds.get()[a_i((unsigned int)(list_idx->second), (unsigned int)i)] = j;
                                    }
                                }
                            }
                        }
                    }
                }
            });
    // save the last computed number of particles
    m_n_ref = n_ref;
    m_n_p = n_p;
    }

}; }; // end namespace freud::bond
