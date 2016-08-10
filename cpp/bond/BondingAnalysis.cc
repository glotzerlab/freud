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

// void BondingAnalysis::reduceArrays()
//     {
//     Index2D transition_indexer = Index2D((m_num_bonds+1), (m_num_bonds+1));
//     // set all values in the transition matrix to 0
//     memset((void*)m_transition_matrix.get(), 0, sizeof(unsigned int)*(m_num_bonds+1)*(m_num_bonds+1));
//     // clear out bond lifetime arrays; don't need to resize cause they are already appropriately sized
//     for (std::vector< std::vector<unsigned int> >::iterator it = m_bond_lifetime_array.begin();
//         it != m_bond_lifetime_array.end(); ++it)
//         {
//         (*it).clear();
//         }
//     for (std::vector< std::vector<unsigned int> >::iterator it = m_overall_lifetime_array.begin();
//         it != m_overall_lifetime_array.end(); ++it)
//         {
//         (*it).clear();
//         }
//     // I do not use parallel reduction in this case due to data structures not being thread safe...
//     // transfer data into bond_lifetime array
//     // for each pidx
//     for (unsigned int i = 0; i < m_num_bonds; i++)
//         {
//         // for each thread local memory
//         for (tbb::enumerable_thread_specific< std::vector< std::vector< unsigned int > > >::const_iterator local_lifetime = m_local_bond_lifetime_array.begin();
//             local_lifetime != m_local_bond_lifetime_array.end(); ++local_lifetime)
//             {
//             for (std::vector< std::vector< unsigned int > >::const_iterator outer_it = (*local_lifetime).begin();
//                 outer_it != (*local_lifetime).end(); ++outer_it)
//                 {
//                 for (std::vector< unsigned int >::const_iterator inner_it = (*outer_it).begin();
//                     inner_it != (*outer_it).end(); ++inner_it)
//                     {
//                     printf("%u ", (*inner_it));
//                     m_bond_lifetime_array[i].push_back((*inner_it));
//                     }
//                 }
//             }
//         }
//     for (unsigned int i = 0; i < m_num_bonds; i++)
//         {
//         std::vector< unsigned int >::iterator it;
//         for (it = m_bond_lifetime_array[i].begin(); it != m_bond_lifetime_array[i].end(); ++it)
//             {
//             printf("%u ", (*it));
//             }
//         printf("\n");
//         }
//     // transfer data into overall_lifetime array
//     // for each pidx
//     for (unsigned int i = 0; i < m_num_bonds; i++)
//         {
//         // for each thread local memory
//         for (tbb::enumerable_thread_specific< std::vector< std::vector< unsigned int > > >::const_iterator local_lifetime = m_local_overall_lifetime_array.begin();
//             local_lifetime != m_local_overall_lifetime_array.end(); ++local_lifetime)
//             {
//             for (std::vector< std::vector< unsigned int > >::const_iterator outer_it = (*local_lifetime).begin();
//                 outer_it != (*local_lifetime).end(); ++outer_it)
//                 {
//                 for (std::vector< unsigned int >::const_iterator inner_it = (*outer_it).begin();
//                     inner_it != (*outer_it).end(); ++inner_it)
//                     {
//                     m_overall_lifetime_array[i].push_back((*inner_it));
//                     }
//                 }
//             }
//         }
//     // transfer into transition matrix
//     for (unsigned int i = 0; i < (m_num_bonds+1); i++)
//         {
//         for (unsigned int j = 0; j < (m_num_bonds+1); j++)
//             {
//             for (tbb::enumerable_thread_specific< unsigned int *>::const_iterator local_transition = m_local_transition_matrix.begin();
//             local_transition != m_local_transition_matrix.end(); ++local_transition)
//                 {
//                 m_transition_matrix.get()[transition_indexer(i, j)] += (*local_transition)[transition_indexer(i, j)];
//                 }
//             }
//         }
//     }

std::vector< std::vector< unsigned int> > BondingAnalysis::getBondLifetimes()
    {
    // if (m_reduce == true)
    //     {
    //     reduceArrays();
    //     }
    // m_reduce = false;
    return m_bond_lifetime_array;
    }

std::vector<unsigned int> BondingAnalysis::getOverallLifetimes()
    {
    // if (m_reduce == true)
    //     {
    //     reduceArrays();
    //     }
    // m_reduce = false;
    return m_overall_lifetime_array;
    }

std::shared_ptr< unsigned int> BondingAnalysis::getTransitionMatrix()
    {
    // if (m_reduce == true)
    //     {
    //     reduceArrays();
    //     }
    // m_reduce = false;
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

void BondingAnalysis::compute(unsigned int* frame0,
                              unsigned int* frame1)
    {
    // track bonds throgh the system
    // for each particle pidx, look at each particle pjdx in a bond pair
    // must create local indexers due to how tbb passes things around
    Index2D transition_indexer = Index2D((m_num_bonds+1), (m_num_bonds+1));
    Index2D m_frame_indexer = Index2D(m_num_bonds, m_num_particles);

    for(unsigned int pidx=0; pidx<m_num_particles; pidx++)
        {
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
        // if (pidx == 5)
        //     {
        //     printf("bonds_0 val = ");
        //     for (std::vector<unsigned int>::iterator v=l_bonds_0.begin(); v!=l_bonds_0.end(); ++v)
        //         {
        //         printf("%u ", (*v));
        //         }
        //     printf("\n");
        //     printf("bonds_1 val = ");
        //     for (std::vector<unsigned int>::iterator v=l_bonds_1.begin(); v!=l_bonds_1.end(); ++v)
        //         {
        //         printf("%u ", (*v));
        //         }
        //     printf("\n");
        //     }
        // iterate through bonds and increment bond lifetime array
        // printf("iterating through bonds\n");
        // // I don't like how I'm handling this...I should do this in the same way I do the over bonding...let's think
        // // that out
        // for(unsigned int bidx=0; bidx<m_num_bonds; bidx++)
        //     {
        //     unsigned int pjdx0 = l_bonds_0[bidx];
        //     unsigned int pjdx1 = l_bonds_1[bidx];
        //     // skip if this is an unbound to unbound transition
        //     if ((pjdx0 == UINT_MAX) && (pjdx1 == UINT_MAX))
        //         {
        //         continue;
        //         }
        //     // create iterator to find things
        //     std::vector< std::pair<unsigned int, unsigned int> >::iterator it;
        //     // get values currently in array
        //     unsigned int current_jdx = m_bond_increment_array[transition_indexer(bidx, pidx)].first;
        //     unsigned int current_count = m_bond_increment_array[transition_indexer(bidx, pidx)].second;
        //     printf("current jdx = %u; current count = %u\n", current_jdx, current_count);
        //     printf("pjdx0 = %u, pjdx1 = %u\n", pjdx0, pjdx1);
        //     // compare, increment as necessary
        //     if ((pjdx0 == UINT_MAX) && (pjdx1 != UINT_MAX))
        //         {
        //         if (current_jdx == UINT_MAX)
        //             {
        //             // add and proceed as usual
        //             m_bond_increment_array[transition_indexer(bidx, pidx)] = std::pair<unsigned int, unsigned int>(pjdx1, 0);
        //             }
        //         else
        //             {
        //             // this shouldn't happen, but let's put an error out for now
        //             printf("local array indicates that a particle is being tracked that should not be tracked\n");
        //             }
        //         }
        //     else if ((pjdx0 != UINT_MAX) && (pjdx1 == UINT_MAX))
        //         {
        //         // look up, add to global array, and delete
        //         if (current_jdx != pjdx0)
        //             {
        //             // this is an error
        //             printf("local array indicated a different particle is in this bond\n");
        //             printf("current_jdx = %u\n", current_jdx);
        //             printf("pjdx0 = %u; pjdx1 = %u\n", pjdx0, pjdx1);
        //             }
        //         else if (current_count == UINT_MAX)
        //             {
        //             printf("count is out of bounds, and indicates that no values were actually counted\n");
        //             }
        //         // increment count array
        //         m_bond_lifetime_array[bidx].push_back(current_count);
        //         // reset values in the increment array
        //         m_bond_increment_array[transition_indexer(bidx, pidx)] = std::pair<unsigned int, unsigned int>(UINT_MAX, UINT_MAX);
        //         }
        //     else
        //         {
        //         // has to be from one particle to another
        //         if (pjdx0 == pjdx1)
        //             {
        //             // increment bond lifetime counter
        //             if (current_jdx >= m_num_particles)
        //                 {
        //                 // data must not have been initialized, so it's the first frame?
        //                 m_bond_increment_array[transition_indexer(bidx, pidx)] = std::pair<unsigned int, unsigned int>(pjdx0, 1);
        //                 }
        //             else
        //                 {
        //                 if (current_jdx != pjdx0)
        //                     {
        //                     printf("wrong particle jdx in local array\n");
        //                     printf("%u vs %u\n", current_jdx, pjdx0);
        //                     printf("UINT_MAX = %u\n", UINT_MAX);
        //                     }
        //                 else if (current_count == UINT_MAX)
        //                     {
        //                     printf("current count is int max, so either the bond lasted too long, or you are incrementing an empty cell\n");
        //                     }
        //                 m_bond_increment_array[transition_indexer(bidx, pidx)].second++;
        //                 }
        //             }
        //         else if (pjdx0 != pjdx1)
        //             {
        //             printf("I should be incrementing...but apparently I am not\n");
        //             // increment count array
        //             m_bond_lifetime_array[bidx].push_back(current_count);
        //             // reset values in the increment array
        //             m_bond_increment_array[transition_indexer(bidx, pidx)] = std::pair<unsigned int, unsigned int>(pjdx1, 0);
        //             }
        //         }
        //     }
        // hmm, now I'm not getting any values to come out of this...or at least that's what it seems
        // for (unsigned int bond_idx = 0; bond_idx < m_num_bonds; bond_idx++)
        //     {
        //     printf("bond %u array size = %lu\n", bond_idx, m_bond_lifetime_array[bond_idx].size());
        //     }
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
        // iterate through and increment transition matrix array
        std::vector<unsigned int>::iterator it_pidx;
        std::vector<std::pair< unsigned int, unsigned int> >::iterator it_pair;
        unsigned int bond_0;
        unsigned int bond_1;
        // if there are bound to bound transitions
        bool dirty_flag = false;
        if (b2b.size() > 0)
            {
            // for each bound to bound transition
            for (std::vector<unsigned int>::iterator it_bond = b2b.begin(); it_bond != b2b.end(); ++it_bond)
                {
                // first increment in the overall bond lifetime array
                it_pair = std::find_if(m_overall_increment_array[pidx].begin(),
                    m_overall_increment_array[pidx].end(), FindParticleIndex(*it_bond));
                if (it_pair != m_overall_increment_array[pidx].end())
                    {
                    // found it, exists, increment
                    (*it_pair).second++;
                    }
                else
                    {
                    // found it, doesn't exist, so create; this should only be for first frame(?)
                    m_overall_increment_array[pidx].push_back(std::pair<unsigned int, unsigned int>(*it_bond, 1));
                    }
                // now increment the transition array
                unsigned int pjdx0;
                unsigned int pjdx1;
                it_pidx = std::find_if(l_bonds_0.begin(), l_bonds_0.end(), FindBondIndex(*it_bond));
                if (it_pidx != l_bonds_0.end())
                    {
                    // the position of pjdx is the bond idx
                    pjdx0 = (*it_pidx);
                    bond_0 = it_pidx-l_bonds_0.begin();
                    }
                else
                    {
                    // this shouldn't happen
                    printf("b2b detected, but bond could not be found\n");
                    }
                it_pidx = std::find_if(l_bonds_1.begin(), l_bonds_1.end(), FindBondIndex(*it_bond));
                if (it_pidx != l_bonds_1.end())
                    {
                    // the position of pjdx is the bond idx
                    pjdx1 = (*it_pidx);
                    bond_1 = it_pidx-l_bonds_1.begin();
                    }
                else
                    {
                    // this shouldn't happen
                    printf("b2b detected, but bond could not be found\n");
                    }
                m_transition_matrix.get()[transition_indexer(bond_0, bond_1)]++;
                // use these bonds to increment bond-specific increment array
                if (bond_0 == bond_1)
                    {
                    // increment the bond-to-bond array
                    // check to make sure that the values are correct
                    if ((m_bond_increment_array[m_frame_indexer(bond_0,pidx)].first != pjdx0) || (m_bond_increment_array[m_frame_indexer(bond_0,pidx)].first != pjdx1))
                        {
                        // sounds like a first frame kind of thing
                        if ((m_bond_increment_array[m_frame_indexer(bond_0,pidx)].first == UINT_MAX) &&
                            (m_bond_increment_array[m_frame_indexer(bond_0,pidx)].first == UINT_MAX))
                            {
                            m_bond_increment_array[m_frame_indexer(bond_1,pidx)] = std::pair<unsigned int, unsigned int>(pjdx1, 1);
                            }
                        else
                            {
                            printf("something went wrong\n");
                            printf("val: (%u, %u) \n", m_bond_increment_array[m_frame_indexer(bond_1,pidx)].first, m_bond_increment_array[m_frame_indexer(bond_1,pidx)].second);
                            }
                        }
                    else
                        {
                        m_bond_increment_array[m_frame_indexer(bond_1,pidx)].second++;
                        }
                    }
                else
                    {
                    // the bond changed; extract count, delete, and create new count
                    unsigned int current_count = m_bond_increment_array[m_frame_indexer(bond_0,pidx)].second;
                    m_bond_lifetime_array[bond_0].push_back(current_count);
                    m_bond_increment_array[m_frame_indexer(bond_0,pidx)] = std::pair<unsigned int, unsigned int>(UINT_MAX, UINT_MAX);
                    m_bond_increment_array[m_frame_indexer(bond_1,pidx)] = std::pair<unsigned int, unsigned int>(pjdx1, 0);
                    }
                }
            }
        if (b2u.size() > 0)
            {
            for (std::vector<unsigned int>::iterator it_bond = b2u.begin(); it_bond != b2u.end(); ++it_bond)
                {
                unsigned int current_count;
                // find, increment, delete
                it_pair = std::find_if(m_overall_increment_array[pidx].begin(),
                    m_overall_increment_array[pidx].end(), FindParticleIndex(*it_bond));
                if (it_pair != m_overall_increment_array[pidx].end())
                    {
                    // found it, exists, get value
                    current_count = (*it_pair).second;
                    // delete old pjdx
                    m_overall_increment_array[pidx].erase(it_pair);
                    // increment
                    m_overall_lifetime_array.push_back(current_count);
                    // find increment the transition matrix
                    it_pidx = std::find_if(l_bonds_0.begin(), l_bonds_0.end(), FindBondIndex(*it_bond));
                    if (it_pidx != l_bonds_0.end())
                        {
                        // the position of pjdx is the bond idx
                        bond_0 = it_pidx-l_bonds_0.begin();
                        }
                    else
                        {
                        // happens on first frame?
                        // this shouldn't happen
                        printf("inner b2u detected, but bond could not be found\n");
                        }
                    bond_1 = m_num_bonds;
                    m_transition_matrix.get()[transition_indexer(bond_0, bond_1)]++;
                    // the bond changed; extract count and stop tracking
                    unsigned int current_count = m_bond_increment_array[m_frame_indexer(bond_0,pidx)].second;
                    if (current_count == UINT_MAX)
                        {
                        dirty_flag = true;
                        printf("UINT_MAX encountered in current count\n");
                        printf("bond idx: %u\n", bond_0);
                        printf("pjdx: %u\n", (*it_bond));
                        printf("pjdx: %u\n", (*it_pidx));
                        printf("val: (%u, %u) \n", m_bond_increment_array[m_frame_indexer(bond_0,pidx)].first, m_bond_increment_array[m_frame_indexer(bond_0,pidx)].second);
                        }
                    m_bond_lifetime_array[bond_0].push_back(current_count);
                    m_bond_increment_array[m_frame_indexer(bond_0,pidx)] = std::pair<unsigned int, unsigned int>(UINT_MAX, UINT_MAX);
                    }
                else
                    {
                    // happens on first frame only
                    // was this happening before?
                    printf("outer b2u detected, but bond could not be found\n");
                    }
                }
            }
        if (u2b.size() > 0)
            {
            for (std::vector<unsigned int>::iterator it_bond = u2b.begin(); it_bond != u2b.end(); ++it_bond)
                {
                // add to the tracking array
                m_overall_increment_array[pidx].push_back(std::pair<unsigned int, unsigned int>(*it_bond, 0));
                // increment the transition array
                unsigned int pjdx1;
                it_pidx = std::find_if(l_bonds_1.begin(), l_bonds_1.end(), FindBondIndex(*it_bond));
                if (it_pidx != l_bonds_1.end())
                    {
                    // the position of pjdx is the bond idx
                    pjdx1 = (*it_pidx);
                    bond_1 = it_pidx-l_bonds_1.begin();
                    }
                else
                    {
                    printf("u2b detected, but bond could not be found\n");
                    }
                bond_0 = m_num_bonds;
                m_transition_matrix.get()[transition_indexer(bond_0, bond_1)]++;
                // bond formed; create start tracking
                m_bond_increment_array[m_frame_indexer(bond_1,pidx)] = std::pair<unsigned int, unsigned int>(pjdx1, 0);
                }
            }
        if (dirty_flag == true)
            {
            printf("pidx: %u\n", pidx);
            printf("b2b: ");
            for (std::vector<unsigned int>::iterator v=b2b.begin(); v!=b2b.end(); ++v)
                {
                printf("%u ", (*v));
                }
            printf("\n");
            printf("b2u: ");
            for (std::vector<unsigned int>::iterator v=b2u.begin(); v!=b2u.end(); ++v)
                {
                printf("%u ", (*v));
                }
            printf("\n");
            printf("u2b: ");
            for (std::vector<unsigned int>::iterator v=u2b.begin(); v!=u2b.end(); ++v)
                {
                printf("%u ", (*v));
                }
            printf("\n");
            }
        }
    m_frame_counter++;
    m_reduce = true;
    }

}; }; // end namespace freud::bond


