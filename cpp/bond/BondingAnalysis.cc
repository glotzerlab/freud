#include "BondingAnalysis.h"
#include "ScopedGILRelease.h"

#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

#include <complex>
#include <map>

using namespace std;
using namespace tbb;

/*! \file EntropicBonding.h
    \brief Compute the hexatic order parameter for each particle
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
    : m_num_particles(num_particles), m_num_bonds(num_bonds), m_frame_counter(0)
    {
    if (m_num_particles < 2)
        throw invalid_argument("must be at least 2 particles to track");
    if (m_num_bonds < 1)
        throw invalid_argument("must be at least 1 bond to track");
    // create arrays to store transition information
    m_transition_matrix = std::shared_ptr<unsigned int>(new unsigned int[(m_num_bonds+1) * (m_num_bonds+1)], std::default_delete<unsigned int[]>());
    memset((void*)m_transition_matrix.get(), 0, sizeof(unsigned int)*(m_num_bonds+1)*(m_num_bonds+1));
    }

// BondingAnalysis::~BondingAnalysis()
//     {
//     for (tbb::enumerable_thread_specific<unsigned int *>::iterator i = m_local_transition_matrix.begin(); i != m_local_transition_matrix.end(); ++i)
//         {
//         delete[] (*i);
//         }
//     }

void BondingAnalysis::reduceArrays()
    {
    Index2D transition_indexer = Index2D((m_num_bonds+1), (m_num_bonds+1));
    // set all values in the transition matrix to 0
    memset((void*)m_transition_matrix.get(), 0, sizeof(unsigned int)*(m_num_bonds+1)*(m_num_bonds+1));
    // clear out bond lifetime arrays; don't need to resize cause they are already appropriately sized
    for (std::vector< std::vector<unsigned int> >::iterator it = m_bond_lifetime_array.begin();
        it != m_bond_lifetime_array.end(); ++it)
        {
        (*it).clear();
        }
    for (std::vector< std::vector<unsigned int> >::iterator it = m_overall_lifetime_array.begin();
        it != m_overall_lifetime_array.end(); ++it)
        {
        (*it).clear();
        }
    // I do not use parallel reduction in this case due to data structures not being thread safe...
    // transfer data into bond_lifetime array
    // for each pidx
    for (unsigned int i = 0; i < m_num_particles; i++)
        {
        // for each thread local memory
        for (tbb::enumerable_thread_specific< std::vector< std::vector< unsigned int > > >::const_iterator local_lifetime = m_local_bond_lifetime_array.begin();
            local_lifetime != m_local_bond_lifetime_array.end(); ++local_lifetime)
            {
            for (std::vector< std::vector< unsigned int > >::const_iterator outer_it = (*local_lifetime).begin();
                outer_it != (*local_lifetime).end(); ++outer_it)
                {
                for (std::vector< unsigned int >::const_iterator inner_it = (*outer_it).begin();
                    inner_it != (*outer_it).end(); ++inner_it)
                    {
                    m_bond_lifetime_array[i].push_back((*inner_it));
                    }
                }
            }
        }
    // transfer data into overall_lifetime array
    // for each pidx
    for (unsigned int i = 0; i < m_num_particles; i++)
        {
        // for each thread local memory
        for (tbb::enumerable_thread_specific< std::vector< std::vector< unsigned int > > >::const_iterator local_lifetime = m_local_overall_lifetime_array.begin();
            local_lifetime != m_local_overall_lifetime_array.end(); ++local_lifetime)
            {
            for (std::vector< std::vector< unsigned int > >::const_iterator outer_it = (*local_lifetime).begin();
                outer_it != (*local_lifetime).end(); ++outer_it)
                {
                for (std::vector< unsigned int >::const_iterator inner_it = (*outer_it).begin();
                    inner_it != (*outer_it).end(); ++inner_it)
                    {
                    m_overall_lifetime_array[i].push_back((*inner_it));
                    }
                }
            }
        }
    // transfer into transition matrix
    for (unsigned int i = 0; i < (m_num_bonds+1); i++)
        {
        for (unsigned int j = 0; j < (m_num_bonds+1); j++)
            {
            for (tbb::enumerable_thread_specific< unsigned int *>::const_iterator local_transition = m_local_transition_matrix.begin();
            local_transition != m_local_transition_matrix.end(); ++local_transition)
                {
                m_transition_matrix.get()[transition_indexer(i, j)] += (*local_transition)[transition_indexer(i, j)];
                }
            }
        }
    }

std::vector< std::vector< unsigned int> > BondingAnalysis::getBondLifetimes()
    {
    if (m_reduce == true)
        {
        reduceArrays();
        }
    m_reduce = false;
    return m_bond_lifetime_array;
    }

std::vector< std::vector< unsigned int> > BondingAnalysis::getOverallLifetimes()
    {
    if (m_reduce == true)
        {
        reduceArrays();
        }
    m_reduce = false;
    return m_overall_lifetime_array;
    }

std::shared_ptr< unsigned int> BondingAnalysis::getTransitionMatrix()
    {
    if (m_reduce == true)
        {
        reduceArrays();
        }
    m_reduce = false;
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

void BondingAnalysis::compute(float* frame0,
                              float* frame1)
    {
    // track bonds throgh the system
    parallel_for(blocked_range<size_t>(0,m_num_particles),
        [=] (const blocked_range<size_t>& br)
            {
            // for each particle pidx, look at each particle pjdx in a bond pair
            // must create local indexers due to how tbb passes things around
            Index2D transition_indexer = Index2D((m_num_bonds+1), (m_num_bonds+1));
            Index2D m_frame_indexer = Index2D(m_num_bonds, m_num_particles);
            // create thread local memory
            bool local_transition_exists;
            m_local_transition_matrix.local(local_transition_exists);
            if (! local_transition_exists)
                {
                m_local_transition_matrix.local() = new unsigned int [(m_num_bonds+1)*(m_num_bonds+1)];
                memset((void*)m_local_transition_matrix.local(), 0, sizeof(unsigned int)*(m_num_bonds+1)*(m_num_bonds+1));
                }
            // I'm not sure if these will be reached, may need to change to if size == 0
            bool local_bond_lifetime;
            m_local_bond_lifetime_array.local(local_bond_lifetime);
            if (! local_bond_lifetime)
                {
                m_local_bond_lifetime_array.local().resize(m_num_bonds);
                }
            bool local_overall_lifetime;
            m_local_overall_lifetime_array.local(local_overall_lifetime);
            if (! local_overall_lifetime)
                {
                m_local_overall_lifetime_array.local().resize(m_num_bonds);
                }
            bool local_bond_increment;
            m_local_bond_increment_array.local(local_bond_increment);
            if (! local_bond_increment)
                {
                m_local_bond_increment_array.local().resize(m_num_particles);
                }
            bool local_overall_increment;
            m_local_overall_increment_array.local(local_overall_increment);
            if (! local_overall_increment)
                {
                m_local_overall_increment_array.local().resize(m_num_particles);
                }

            for(size_t i=br.begin(); i!=br.end(); ++i)
                {
                // create local vectors to store bonding information
                // specifically which particles pjdx are in which bonds with particles pidx
                std::vector<unsigned int> l_bonds_0;
                std::vector<unsigned int> l_bonds_1;
                l_bonds_0.resize(m_num_bonds);
                l_bonds_1.resize(m_num_bonds);
                std::vector<unsigned int> s_bonds_0;
                std::vector<unsigned int> s_bonds_1;
                s_bonds_0.resize(m_num_bonds);
                s_bonds_1.resize(m_num_bonds);
                // populate bond vector
                for(unsigned int j=0; j<m_num_bonds; j++)
                    {
                    unsigned int pjdx0 = frame0[m_frame_indexer(j, i)];
                    unsigned int pjdx1 = frame1[m_frame_indexer(j, i)];
                    l_bonds_0[j] = pjdx0;
                    l_bonds_1[j] = pjdx1;
                    if (pjdx0 != UINT_MAX)
                        s_bonds_0[j] = frame0[m_frame_indexer(j, i)];
                    if (pjdx1 != UINT_MAX)
                        s_bonds_1[j] = frame1[m_frame_indexer(j, i)];
                    }
                // sort bonds
                std::sort(s_bonds_0.begin(), s_bonds_0.end());
                std::sort(s_bonds_1.begin(), s_bonds_1.end());
                // iterate through bonds and increment bond lifetime array
                for(unsigned int j=0; j<m_num_bonds; j++)
                    {
                    unsigned int pjdx0 = l_bonds_0[j];
                    unsigned int pjdx1 = l_bonds_1[j];
                    // create iterator to find things
                    std::vector< std::pair<unsigned int, unsigned int> >::iterator it;
                    // compare, increment as necessary
                    // skip if this is an unbound to unbound transition
                    if ((pjdx0 == UINT_MAX) && (pjdx1 == UINT_MAX))
                        {
                        continue;
                        }
                    else if ((pjdx0 == UINT_MAX) && (pjdx1 != UINT_MAX))
                        {
                        // create and increment
                        m_local_bond_increment_array.local()[j].push_back(std::pair<unsigned int, unsigned int>(pjdx1, 0));
                        // I don't think this is right
                        // unsigned int bond_0 = m_num_bonds;
                        // unsigned int bond_1 = j;
                        // ++m_local_transition_matrix.local()[transition_indexer(bond_0, bond_1)];
                        }
                    else if ((pjdx0 != UINT_MAX) && (pjdx1 == UINT_MAX))
                        {
                        // look up, add to global array, and delete
                        unsigned int current_count;
                        it = std::find_if(m_local_bond_increment_array.local()[j].begin(),
                            m_local_bond_increment_array.local()[j].end(), FindParticleIndex(pjdx0));
                        if (it != m_local_bond_increment_array.local()[j].end())
                            {
                            // found it, exists, get value
                            current_count = (*it).second;
                            // delete old pjdx
                            m_local_bond_increment_array.local()[j].erase(it);
                            }
                        else
                            {
                            // THIS DOESN'T MAKE SENSE
                            // found it, doesn't exist, so create; this should only be for first frame(?)
                            current_count = 0;
                            }
                        // add to array
                        m_local_bond_lifetime_array.local()[i].push_back(current_count);
                        }
                    else
                        {
                        // has to be from one particle to another
                        if (pjdx0 == pjdx1)
                            {
                            // increment bond lifetime counter
                            // search to find in vector
                            it = std::find_if(m_local_bond_increment_array.local()[j].begin(),
                                m_local_bond_increment_array.local()[j].end(), FindParticleIndex(pjdx0));
                            if (it != m_local_bond_increment_array.local()[j].end())
                                {
                                // found it, exists, increment
                                (*it).second++;
                                }
                            else
                                {
                                // found it, doesn't exist, so create; this should only be for first frame(?)
                                m_local_bond_increment_array.local()[j].push_back(std::pair<unsigned int, unsigned int>(pjdx1, 1));
                                }
                            }
                        else if (pjdx0 != pjdx1)
                            {
                            // get current count
                            unsigned int current_count;
                            it = std::find_if(m_local_bond_increment_array.local()[j].begin(),
                                m_local_bond_increment_array.local()[j].end(), FindParticleIndex(pjdx0));
                            if (it != m_local_bond_increment_array.local()[j].end())
                                {
                                // found it, exists, get value
                                current_count = (*it).second;
                                // delete old pjdx
                                m_local_bond_increment_array.local()[j].erase(it);
                                }
                            else
                                {
                                // this shouldn't happen?
                                current_count = 0;
                                }
                            // add to array
                            m_local_bond_lifetime_array.local()[i].push_back(current_count);
                            // add new bond to track
                            m_local_bond_increment_array.local()[j].push_back(std::pair<unsigned int, unsigned int>(pjdx1, 0));
                            }
                        }
                    }
                // find and iterate overall bond lifetimes
                // find all pjdx in frame 0 not currently being tracked
                // is this frame 0 only
                // then, for all pjdx being tracked, find which ones aren't present in the next frame
                // I should be able to do the transition matrix at the same time
                std::vector<unsigned int> b2b;
                std::vector<unsigned int> b2u;
                std::vector<unsigned int> u2b;
                std::set_intersection(s_bonds_0.begin(), s_bonds_0.end(), s_bonds_1.begin(), s_bonds_1.end(), std::back_inserter(b2b));
                std::set_difference(s_bonds_0.begin(), s_bonds_0.end(), s_bonds_1.begin(), s_bonds_1.end(), std::back_inserter(b2u));
                std::set_difference(s_bonds_1.begin(), s_bonds_1.end(), s_bonds_0.begin(), s_bonds_0.end(), std::back_inserter(u2b));
                // iterate through and increment transition matrix array
                std::vector<unsigned int>::iterator it_pidx;
                std::vector<std::pair< unsigned int, unsigned int> >::iterator it_pair;
                unsigned int bond_0;
                unsigned int bond_1;
                for (std::vector<unsigned int>::iterator it_bond = b2b.begin(); it_bond != b2b.end(); ++it_bond)
                    {
                    // first increment in the overall bond lifetime array
                    it_pair = std::find_if(m_local_overall_increment_array.local()[i].begin(),
                        m_local_overall_increment_array.local()[i].end(), FindParticleIndex(*it_bond));
                    if (it_pair != m_local_overall_increment_array.local()[i].end())
                        {
                        // found it, exists, increment
                        (*it_pair).second++;
                        }
                    else
                        {
                        // found it, doesn't exist, so create; this should only be for first frame(?)
                        m_local_overall_increment_array.local()[i].push_back(std::pair<unsigned int, unsigned int>(*it_bond, 1));
                        }
                    // now increment the transition array
                    it_pidx = std::find_if(l_bonds_0.begin(), l_bonds_0.end(), FindBondIndex(*it_bond));
                    if (it_pidx != l_bonds_0.end())
                        {
                        // the position of pjdx is the bond idx
                        bond_0 = it_pidx-l_bonds_0.begin();
                        }
                    else
                        {
                        // this shouldn't happen
                        continue;
                        }
                    it_pidx = std::find_if(l_bonds_1.begin(), l_bonds_1.end(), FindBondIndex(*it_bond));
                    if (it_pidx != l_bonds_1.end())
                        {
                        // the position of pjdx is the bond idx
                        bond_1 = it_pidx-l_bonds_1.begin();
                        }
                    else
                        {
                        // this shouldn't happen
                        continue;
                        }
                    ++m_local_transition_matrix.local()[transition_indexer(bond_0, bond_1)];
                    }
                for (std::vector<unsigned int>::iterator it_bond = b2u.begin(); it_bond != b2u.end(); ++it_bond)
                    {
                    unsigned int current_count;
                    // find, increment, delete
                    it_pair = std::find_if(m_local_overall_increment_array.local()[i].begin(),
                        m_local_overall_increment_array.local()[i].end(), FindParticleIndex(*it_bond));
                    if (it_pair != m_local_overall_increment_array.local()[i].end())
                        {
                        // found it, exists, get value
                        current_count = (*it_pair).second;
                        // delete old pjdx
                        m_local_overall_increment_array.local()[i].erase(it_pair);
                        }
                    else
                        {
                        // shouldn't happen
                        ;
                        }
                    // increment
                    m_local_overall_lifetime_array.local()[i].push_back(current_count);
                    // find increment the transition matrix
                    it_pidx = std::find_if(l_bonds_0.begin(), l_bonds_0.end(), FindBondIndex(*it_bond));
                    if (it_pidx != l_bonds_0.end())
                        {
                        // the position of pjdx is the bond idx
                        bond_0 = it_pidx-l_bonds_0.begin();
                        }
                    else
                        {
                        // this shouldn't happen
                        continue;
                        }
                    bond_1 = m_num_bonds;
                    ++m_local_transition_matrix.local()[transition_indexer(bond_0, bond_1)];
                    }
                for (std::vector<unsigned int>::iterator it_bond = u2b.begin(); it_bond != u2b.end(); ++it_bond)
                    {
                    // add to the tracking array
                    m_local_overall_increment_array.local()[i].push_back(std::pair<unsigned int, unsigned int>(*it_bond, 0));
                    // increment the transition array
                    it_pidx = std::find_if(l_bonds_1.begin(), l_bonds_1.end(), FindBondIndex(*it_bond));
                    if (it_pidx != l_bonds_1.end())
                        {
                        // the position of pjdx is the bond idx
                        bond_1 = it_pidx-l_bonds_1.begin();
                        }
                    else
                        {
                        // this shouldn't happen
                        continue;
                        }
                    bond_0 = m_num_bonds;
                    ++m_local_transition_matrix.local()[transition_indexer(bond_0, bond_1)];
                    }
                }
            });
    m_frame_counter++;
    m_reduce = true;
    }

}; }; // end namespace freud::bond


