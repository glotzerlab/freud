// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <complex>
#include <map>
#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include "BondingAnalysis.h"

using namespace std;
using namespace tbb;

/*! \file BondingAnalysis.cc
    \brief Determines the bond lifetimes and flux present in the system.
*/

namespace freud { namespace bond {

struct FindParticleIndex
    {
    FindParticleIndex(unsigned int pjdx)
    : m_pjdx(pjdx)
    {
    }

    unsigned int m_pjdx;
    bool operator()
    ( const std::pair<unsigned int, unsigned int> &p )
        {
        return (m_pjdx == p.first);
        }
    };

struct FindBondIndex
    {
    FindBondIndex(unsigned int pjdx)
    : m_pjdx(pjdx)
    {
    }

    unsigned int m_pjdx;
    bool operator()
    ( const unsigned int &p )
        {
        return (m_pjdx == p);
        }
    };

BondingAnalysis::BondingAnalysis(unsigned int num_particles,
                                 unsigned int num_bonds)
    : m_num_particles(num_particles), m_num_bonds(num_bonds), m_frame_counter(0), m_reduce(true)
    {
    if (m_num_particles < 2)
        throw invalid_argument("must be at least 2 particles to track");
    if (m_num_bonds < 1)
        throw invalid_argument("must be at least 1 bond to track");
    // create arrays to store transition information
    m_transition_matrix = std::shared_ptr<unsigned int>(new unsigned int[(m_num_bonds+1) * (m_num_bonds+1)], std::default_delete<unsigned int[]>());
    m_bond_increment_array = new std::pair<unsigned int, unsigned int> [m_num_bonds*m_num_particles];
    for (unsigned int i = 0; i < (m_num_bonds*m_num_particles); i++)
        {
        m_bond_increment_array[i] = std::pair<unsigned int, unsigned int>(UINT_MAX, UINT_MAX);
        }
    m_overall_increment_array.resize(m_num_particles);
    m_bond_lifetime_array.resize(m_num_bonds);
    m_overall_lifetime_array.resize(0);
    memset((void*)m_transition_matrix.get(), 0, sizeof(unsigned int)*(m_num_bonds+1)*(m_num_bonds+1));
    }

BondingAnalysis::~BondingAnalysis()
    {
    delete m_bond_increment_array;
    }

std::vector< std::vector< unsigned int> > BondingAnalysis::getBondLifetimes()
    {
    return m_bond_lifetime_array;
    }

std::vector<unsigned int> BondingAnalysis::getOverallLifetimes()
    {
    return m_overall_lifetime_array;
    }

std::shared_ptr< unsigned int> BondingAnalysis::getTransitionMatrix()
    {
    return m_transition_matrix;
    }

unsigned int BondingAnalysis::getNumFrames()
    {
    return m_frame_counter;
    }

unsigned int BondingAnalysis::getNumParticles()
    {
    return m_num_particles;
    }

unsigned int BondingAnalysis::getNumBonds()
    {
    return m_num_particles;
    }

void BondingAnalysis::initialize(unsigned int* frame0)
    {
    Index2D m_frame_indexer = Index2D(m_num_bonds, m_num_particles);

    // reset all values
    for (std::vector< std::vector< std::pair<unsigned int, unsigned int> > >::iterator it = m_overall_increment_array.begin();
        it != m_overall_increment_array.end(); ++it)
        {
        (*it).clear();
        }
    for (unsigned int i = 0; i < (m_num_bonds*m_num_particles); i++)
        {
        m_bond_increment_array[i] = std::pair<unsigned int, unsigned int>(UINT_MAX, UINT_MAX);
        }

    for(unsigned int pidx=0; pidx<m_num_particles; pidx++)
        {
        // for each particle
        std::vector<unsigned int> l_bonds_0;
        l_bonds_0.resize(m_num_bonds);
        std::vector<unsigned int> s_bonds_0;
        s_bonds_0.resize(0);
        // populate bond vectors
        for(unsigned int bidx=0; bidx<m_num_bonds; bidx++)
            {
            // put all bonds in either the full list, or the bond-only list
            unsigned int pjdx0 = frame0[m_frame_indexer(bidx, pidx)];
            l_bonds_0[bidx] = pjdx0;
            if (pjdx0 != UINT_MAX)
                s_bonds_0.push_back(frame0[m_frame_indexer(bidx, pidx)]);
            }
        for (std::vector<unsigned int>::iterator it_pjdx=s_bonds_0.begin(); it_pjdx!=s_bonds_0.end(); ++it_pjdx)
            {
            // for each bound particle
            std::vector<unsigned int>::iterator it_bond;
            it_bond = std::find_if(l_bonds_0.begin(), l_bonds_0.end(), FindBondIndex(*it_pjdx));
            if (it_bond != l_bonds_0.end())
                {
                // if the bond is found
                unsigned int bidx = it_bond - l_bonds_0.begin();
                // add bond to the overall array
                m_overall_increment_array[pidx].push_back(std::pair<unsigned int, unsigned int>((*it_pjdx), 0));
                // add bond to the bond array
                m_bond_increment_array[m_frame_indexer(bidx, pidx)] = std::pair<unsigned int, unsigned int>((*it_pjdx), 0);
                }
            else
                {
                // this should not be reached
                printf("unknown bond found during initialization\n");
                }
            }
        }
    }

void BondingAnalysis::compute(unsigned int* frame0,
                              unsigned int* frame1)
    {
    // track bonds throgh the system
    Index2D transition_indexer = Index2D((m_num_bonds+1), (m_num_bonds+1));
    Index2D m_frame_indexer = Index2D(m_num_bonds, m_num_particles);

    for(unsigned int pidx=0; pidx<m_num_particles; pidx++)
        {
        // for each particle pidx, look at each particle pjdx in a bond pair
        // create local vectors to store bonding information
        // specifically which particles pjdx are in which bonds with particles pidx
        std::vector<unsigned int> l_bonds_0;
        std::vector<unsigned int> l_bonds_1;
        l_bonds_0.resize(m_num_bonds);
        l_bonds_1.resize(m_num_bonds);
        std::vector<unsigned int> s_bonds_0;
        std::vector<unsigned int> s_bonds_1;
        s_bonds_0.resize(0);
        s_bonds_1.resize(0);
        // populate bond vectors
        for(unsigned int bidx=0; bidx<m_num_bonds; bidx++)
            {
            unsigned int pjdx0 = frame0[m_frame_indexer(bidx, pidx)];
            unsigned int pjdx1 = frame1[m_frame_indexer(bidx, pidx)];
            l_bonds_0[bidx] = pjdx0;
            l_bonds_1[bidx] = pjdx1;
            if (pjdx0 != UINT_MAX)
                s_bonds_0.push_back(frame0[m_frame_indexer(bidx, pidx)]);
            if (pjdx1 != UINT_MAX)
                s_bonds_1.push_back(frame1[m_frame_indexer(bidx, pidx)]);
            }
        // sort bonds for later
        std::sort(s_bonds_0.begin(), s_bonds_0.end());
        std::sort(s_bonds_1.begin(), s_bonds_1.end());
        // create vectors to track bound to bound, bound to unbound, and unbound to bound particles
        std::vector<unsigned int> b2b;
        std::vector<unsigned int> b2u;
        std::vector<unsigned int> u2b;
        b2b.resize(0);
        b2u.resize(0);
        u2b.resize(0);
        // use intersections and differences to determine these
        std::set_intersection(s_bonds_0.begin(), s_bonds_0.end(), s_bonds_1.begin(), s_bonds_1.end(), std::back_inserter(b2b));
        std::set_difference(s_bonds_0.begin(), s_bonds_0.end(), s_bonds_1.begin(), s_bonds_1.end(), std::back_inserter(b2u));
        std::set_difference(s_bonds_1.begin(), s_bonds_1.end(), s_bonds_0.begin(), s_bonds_0.end(), std::back_inserter(u2b));
        // bound to unbound transition
        if (b2u.size() > 0)
            {
            for (std::vector<unsigned int>::iterator it_pjdx = b2u.begin(); it_pjdx != b2u.end(); ++it_pjdx)
                {
                unsigned int bond_0 = UINT_MAX;
                unsigned int bond_1 = m_num_bonds;
                // find, increment, delete
                std::vector<std::pair< unsigned int, unsigned int> >::iterator it_pair;
                it_pair = std::find_if(m_overall_increment_array[pidx].begin(),
                    m_overall_increment_array[pidx].end(), FindParticleIndex(*it_pjdx));
                if (it_pair != m_overall_increment_array[pidx].end())
                    {
                    // bond found
                    std::vector<unsigned int>::iterator it_bond;
                    // found it, exists, get value
                    unsigned int current_count = (*it_pair).second;
                    // delete old pjdx
                    m_overall_increment_array[pidx].erase(it_pair);
                    // increment
                    m_overall_lifetime_array.push_back(current_count);
                    // find increment the transition matrix
                    it_bond = std::find_if(l_bonds_0.begin(), l_bonds_0.end(), FindBondIndex(*it_pjdx));
                    if (it_bond != l_bonds_0.end())
                        bond_0 = it_bond-l_bonds_0.begin();
                    m_transition_matrix.get()[transition_indexer(bond_0, bond_1)]++;
                    // the bond changed; extract count and stop tracking
                    current_count = m_bond_increment_array[m_frame_indexer(bond_0,pidx)].second;
                    if ((current_count != 0) && (current_count != UINT_MAX))
                        m_bond_lifetime_array[bond_0].push_back(current_count);
                    // let's check to make sure the increment array matches
                    if (m_bond_increment_array[m_frame_indexer(bond_0,pidx)].first == (*it_bond))
                        m_bond_increment_array[m_frame_indexer(bond_0,pidx)] = std::pair<unsigned int, unsigned int>(UINT_MAX, UINT_MAX);
                    }
                }
            }
        if (b2b.size() > 0)
            {
            // for each bound to bound transition
            for (std::vector<unsigned int>::iterator it_pjdx=b2b.begin(); it_pjdx!=b2b.end(); ++it_pjdx)
                {
                unsigned int bond_0 = UINT_MAX;
                unsigned int bond_1 = UINT_MAX;
                // find, increment, delete
                std::vector<std::pair< unsigned int, unsigned int> >::iterator it_pair;
                // first increment in the overall bond lifetime array
                it_pair = std::find_if(m_overall_increment_array[pidx].begin(),
                    m_overall_increment_array[pidx].end(), FindParticleIndex(*it_pjdx));
                if (it_pair != m_overall_increment_array[pidx].end())
                    (*it_pair).second++;
                // now increment the transition array
                std::vector<unsigned int>::iterator it_bond;
                it_bond = std::find_if(l_bonds_0.begin(), l_bonds_0.end(), FindBondIndex(*it_pjdx));
                if (it_bond != l_bonds_0.end())
                    bond_0 = it_bond-l_bonds_0.begin();
                it_bond = std::find_if(l_bonds_1.begin(), l_bonds_1.end(), FindBondIndex(*it_pjdx));
                if (it_bond != l_bonds_1.end())
                    bond_1 = it_bond-l_bonds_1.begin();
                if ((bond_0 != UINT_MAX) && (bond_1 != UINT_MAX))
                    m_transition_matrix.get()[transition_indexer(bond_0, bond_1)]++;
                // use these bonds to increment bond-specific increment array
                if ((bond_0 != UINT_MAX) && (bond_1 != UINT_MAX))
                    {
                    if (bond_0 == bond_1)
                        {
                        // increment the bond-to-bond array
                        // check to make sure that the values are correct
                        if ((m_bond_increment_array[m_frame_indexer(bond_0,pidx)].first != (*it_pjdx)) || (m_bond_increment_array[m_frame_indexer(bond_1,pidx)].first != (*it_pjdx)))
                            {
                            // looks like this line is never reached, leaving in for now
                            printf("huh?\n");
                            // sounds like a first frame kind of thing
                            if ((m_bond_increment_array[m_frame_indexer(bond_0,pidx)].first == UINT_MAX) &&
                                (m_bond_increment_array[m_frame_indexer(bond_1,pidx)].first == UINT_MAX))
                                {
                                // this line is being reached
                                printf("uninitialized memory detected\n");
                                m_bond_increment_array[m_frame_indexer(bond_1,pidx)] = std::pair<unsigned int, unsigned int>((*it_pjdx), 1);
                                }
                            }
                        else
                            m_bond_increment_array[m_frame_indexer(bond_1,pidx)].second++;
                        }
                    else
                        {
                        // the bond changed; extract count
                        unsigned int current_count = m_bond_increment_array[m_frame_indexer(bond_0,pidx)].second;
                        // add count to the lifetime array
                        if ((current_count != 0) && (current_count != UINT_MAX))
                            m_bond_lifetime_array[bond_0].push_back(current_count);
                        // delete from array only if pjdx matches
                        if (m_bond_increment_array[m_frame_indexer(bond_0,pidx)].first == (*it_pjdx))
                            m_bond_increment_array[m_frame_indexer(bond_0,pidx)] = std::pair<unsigned int, unsigned int>(UINT_MAX, UINT_MAX);
                        // ensure we do not overwrite other data
                        if (m_bond_increment_array[m_frame_indexer(bond_1,pidx)].first != (*it_pjdx))
                            {
                            // there is particle data that needs to be saved
                            current_count = m_bond_increment_array[m_frame_indexer(bond_1,pidx)].second;
                            if ((current_count != 0) && (current_count != UINT_MAX))
                                m_bond_lifetime_array[bond_1].push_back(current_count);
                            }
                        m_bond_increment_array[m_frame_indexer(bond_1,pidx)] = std::pair<unsigned int, unsigned int>((*it_pjdx), 0);
                        }
                    }
                }
            }
        if (u2b.size() > 0)
            {
            for (std::vector<unsigned int>::iterator it_pjdx = u2b.begin(); it_pjdx != u2b.end(); ++it_pjdx)
                {
                unsigned int bond_0 = m_num_bonds;
                unsigned int bond_1 = UINT_MAX;
                // add to the tracking array
                m_overall_increment_array[pidx].push_back(std::pair<unsigned int, unsigned int>((*it_pjdx), 0));
                // increment the transition array
                std::vector<unsigned int>::iterator it_bond;
                it_bond = std::find_if(l_bonds_1.begin(), l_bonds_1.end(), FindBondIndex(*it_pjdx));
                if (it_bond != l_bonds_1.end())
                    {
                    // the position of pjdx is the bond idx
                    bond_1 = it_bond-l_bonds_1.begin();
                    m_transition_matrix.get()[transition_indexer(bond_0, bond_1)]++;
                    // bond formed; create start tracking
                    m_bond_increment_array[m_frame_indexer(bond_1,pidx)] = std::pair<unsigned int, unsigned int>((*it_pjdx), 0);
                    }
                }
            }
        }
    m_frame_counter++;
    m_reduce = true;
    }

}; }; // end namespace freud::bond
