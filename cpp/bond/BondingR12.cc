// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <complex>
#include <map>
#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include "BondingR12.h"

using namespace std;
using namespace tbb;

namespace freud { namespace bond {

BondingR12::BondingR12(float r_max,
                       unsigned int n_r,
                       unsigned int n_t2,
                       unsigned int n_t1,
                       unsigned int n_bonds,
                       unsigned int *bond_map,
                       unsigned int *bond_list)
    : m_box(box::Box()), m_r_max(r_max), m_t_max(2.0*M_PI), m_nbins_r(n_r), m_nbins_t2(n_t2), m_nbins_t1(n_t1),
      m_n_bonds(n_bonds), m_bond_map(bond_map), m_bond_list(bond_list), m_n_ref(0), m_n_p(0)
    {
    // create the unsigned int array to store whether or not a particle is paired
    m_bonds = std::shared_ptr<unsigned int>(new unsigned int[m_n_ref*m_n_bonds], std::default_delete<unsigned int[]>());
    if (m_nbins_r < 1)
        throw invalid_argument("must be at least 1 bin in r");
    if (m_nbins_t1 < 1)
        throw invalid_argument("must be at least 1 bin in T1");
    if (m_nbins_t2 < 1)
        throw invalid_argument("must be at least 1 bin in T2");
    if (m_r_max < 0.0f)
        throw invalid_argument("rmax must be positive");
    if (m_n_bonds < 1)
        throw invalid_argument("must have at least 1 bond");
    // calculate dx, dy
    m_dr = m_r_max / float(m_nbins_r);
    m_dt1 = m_t_max / float(m_nbins_t1);
    m_dt2 = m_t_max / float(m_nbins_t2);
    if (m_dr > m_r_max)
        throw invalid_argument("rmax must be greater than dr");
    if (m_dt1 > m_t_max)
        throw invalid_argument("tmax must be greater than dt1");
    if (m_dt2 > m_t_max)
        throw invalid_argument("tmax must be greater than dt2");

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
    }

BondingR12::~BondingR12()
    {
    }

std::shared_ptr<unsigned int> BondingR12::getBonds()
    {
    return m_bonds;
    }

std::map<unsigned int, unsigned int> BondingR12::getListMap()
    {
    return m_list_map;
    }

std::map<unsigned int, unsigned int> BondingR12::getRevListMap()
    {
    return m_rev_list_map;
    }

void BondingR12::compute(box::Box& box,
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
            float dr_inv = 1.0f / m_dr;
            float dt1_inv = 1.0f / m_dt1;
            float dt2_inv = 1.0f / m_dt2;
            float rmaxsq = m_r_max * m_r_max;
            // indexer for bond list
            Index2D a_i = Index2D(m_n_bonds, n_ref);
            // indexer for bond map
            Index3D b_i = Index3D(m_nbins_t1, m_nbins_t2, m_nbins_r);

            size_t bond(nlist->find_first_index(br.begin()));
            for(size_t i=br.begin(); i!=br.end(); ++i)
                {
                // get position, orientation of particle i
                vec3<float> ref_pos = ref_points[i];
                float ref_angle = ref_orientations[i];

                for(; bond < nlist->getNumBonds() && neighbor_list[2*bond] == i; ++bond)
                    {
                    const size_t j(neighbor_list[2*bond + 1]);
                        {
                        //compute r between the two particles
                        vec3<float> delta = m_box.wrap(points[j] - ref_pos);

                        float rsq = dot(delta, delta);
                        // particle cannot pair with itself...i != j is probably better?
                        if ((rsq < 1e-6) || (rsq > rmaxsq))
                            {
                            continue;
                            }
                        // determine which histogram bin to look in
                        float r = sqrtf(rsq);
                        float d_theta1 = atan2(delta.y, delta.x);
                        float d_theta2 = atan2(-delta.y, -delta.x);
                        float t1 = ref_angle - d_theta1;
                        float t2 = orientations[j] - d_theta2;
                        // make sure that t1, t2 are bounded between 0 and 2PI
                        t1 = fmod(t1, 2*M_PI);
                        if (t1 < 0)
                            {
                            t1 += 2*M_PI;
                            }
                        t2 = fmod(t2, 2*M_PI);
                        if (t2 < 0)
                            {
                            t2 += 2*M_PI;
                            }
                        // bin that point
                        float bin_r = r * dr_inv;
                        float bin_t1 = floorf(t1 * dt1_inv);
                        float bin_t2 = floorf(t2 * dt2_inv);
                        // fast float to int conversion with truncation
                        #ifdef __SSE2__
                        unsigned int ibin_r = _mm_cvtt_ss2si(_mm_load_ss(&bin_r));
                        unsigned int ibin_t1 = _mm_cvtt_ss2si(_mm_load_ss(&bin_t1));
                        unsigned int ibin_t2 = _mm_cvtt_ss2si(_mm_load_ss(&bin_t2));
                        #else
                        unsigned int ibin_r = (unsigned int)(bin_r);
                        unsigned int ibin_t1 = (unsigned int)(bin_t1);
                        unsigned int ibin_t2 = (unsigned int)(bin_t2);
                        #endif

                        // log the bond
                        if ((ibin_r < m_nbins_r) && (ibin_t1 < m_nbins_t1) && (ibin_t2 < m_nbins_t2))
                            {
                            // find the bond that corresponds to this point
                            unsigned int bond = m_bond_map[b_i(ibin_t1, ibin_t2, ibin_r)];
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
            });
    m_n_ref = n_ref;
    m_n_p = n_p;
    }

}; }; // end namespace freud::bond
