// Copyright (c) 2010-2016 The Regents of the University of Michigan
// This file is part of the Freud project, released under the BSD 3-Clause License.

#include "BondingXY2D.h"
#include "ScopedGILRelease.h"

#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include <complex>
#include <map>

using namespace std;
using namespace tbb;

namespace freud { namespace bond {

struct FindParticle
    {
    FindParticle(unsigned int pjdx)
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
    // m_bonds = std::shared_ptr<unsigned int>(new unsigned int[m_n_ref*m_n_bonds], std::default_delete<unsigned int[]>());
    m_bond_tracker_array.resize(m_n_ref);
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

    // create cell list
    m_lc = new locality::LinkCell(m_box, m_r_max);
    }

BondingXY2D::~BondingXY2D()
    {
    delete m_lc;
    }

// std::shared_ptr<unsigned int> BondingXY2D::getBonds()
    // {
    // return m_bonds;
    // }

std::vector<unsigned int> BondingXY2D::getBondLifetimes()
    {
    return m_bond_lifetime_array;
    }

std::vector<unsigned int> BondingXY2D::getOverallLifetimes()
    {
    return m_overall_lifetime_array;
    }

std::map<unsigned int, unsigned int> BondingXY2D::getListMap()
    {
    return m_list_map;
    }

std::map<unsigned int, unsigned int> BondingXY2D::getRevListMap()
    {
    return m_rev_list_map;
    }

void BondingXY2D::initialize(box::Box& box,
                             vec3<float> *ref_points,
                             float *ref_orientations,
                             unsigned int n_ref,
                             vec3<float> *points,
                             float *orientations,
                             unsigned int n_p)
    {
    m_bond_tracker_array.resize(n_ref);
    m_box = box;
    // compute the cell list
    m_lc->computeCellList(m_box,points,n_p);
    parallel_for(blocked_range<size_t>(0,n_ref),
        [=] (const blocked_range<size_t>& br)
            {
            float dx_inv = 1.0f / m_dx;
            float dy_inv = 1.0f / m_dy;
            float rmaxsq = m_r_max * m_r_max;
            // indexer for bond list
            Index2D a_i = Index2D(m_n_bonds, n_ref);
            // indexer for bond map
            Index2D b_i = Index2D(m_nbins_x, m_nbins_y);

            for(size_t i=br.begin(); i!=br.end(); ++i)
                {
                std::vector<unsigned int> touched_pjdxs;
                touched_pjdxs.resize(0);
                // get position, orientation of particle i
                vec3<float> ref_pos = ref_points[i];
                float ref_angle = ref_orientations[i];
                // get cell for particle i
                unsigned int ref_cell = m_lc->getCell(ref_pos);

                //loop over neighbor cells
                const std::vector<unsigned int>& neigh_cells = m_lc->getCellNeighbors(ref_cell);
                for (unsigned int neigh_idx = 0; neigh_idx < neigh_cells.size(); neigh_idx++)
                    {
                    // get neighbor cell
                    unsigned int neigh_cell = neigh_cells[neigh_idx];

                    // iterate over the particles in that cell
                    locality::LinkCell::iteratorcell it = m_lc->itercell(neigh_cell);
                    for (unsigned int j = it.next(); !it.atEnd(); j=it.next())
                        {
                        //compute r between the two particles
                        vec3<float> delta = m_box.wrap(points[j] - ref_pos);

                        float rsq = dot(delta, delta);
                        // particle cannot pair with itself...i != j is probably better?
                        if ((i == j) || (rsq > rmaxsq))
                            // this should skip i == j and if rsq would segfault in the bond map lookup
                            // and the set difference should catch the b2u
                            {
                            continue;
                            }
                        // determine which histogram bin to look in
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
                        unsigned int ibin_x = _mm_cvtt_ss2si(_mm_load_ss(&binx));
                        unsigned int ibin_y = _mm_cvtt_ss2si(_mm_load_ss(&biny));
                        #else
                        unsigned int ibin_x = (unsigned int)(binx);
                        unsigned int ibin_y = (unsigned int)(biny);
                        #endif


                        // find the bond
                        // bin if bond is tracked
                        bool isBondTracked = false;
                        unsigned int bond;
                        if ((ibin_x < m_nbins_x) && (ibin_y < m_nbins_y))
                            {
                            // find the bond that corresponds to this point
                            bond = m_bond_map[b_i(ibin_x, ibin_y)];
                            // get the index from the map
                            auto list_idx = m_list_map.find(bond);
                            if (list_idx != m_list_map.end())
                                {
                                isBondTracked = true;
                                }
                            }
                        if (!isBondTracked)
                            {
                            continue;
                            }
                        std::vector<unsigned int>::iterator insert_idx = std::upper_bound(touched_pjdxs.begin(), touched_pjdxs.end(), j);
                        auto l_index = std::distance(touched_pjdxs.begin(), insert_idx);
                        touched_pjdxs.insert(insert_idx, j);

                        std::pair< unsigned int, std::vector<unsigned int> > new_pjdx;
                        std::vector<unsigned int> new_element;
                        new_element.resize(0);
                        // bond label
                        new_element.push_back(bond);
                        // bond lifetime count
                        new_element.push_back(1);
                        // overall lifetime count
                        new_element.push_back(1);
                        new_pjdx.first = j;
                        new_pjdx.second = new_element;
                        // use insertion sort
                        m_bond_tracker_array[i].insert(m_bond_tracker_array[i].begin()+l_index, new_pjdx);
                        }
                    }
                }
            });
    m_n_ref = n_ref;
    m_n_p = n_p;
    }


void BondingXY2D::compute(box::Box& box,
                          vec3<float> *ref_points,
                          float *ref_orientations,
                          unsigned int n_ref,
                          vec3<float> *points,
                          float *orientations,
                          unsigned int n_p)
    {
    m_box = box;
    // compute the cell list
    m_lc->computeCellList(m_box,points,n_p);
    // compute the order parameter
    parallel_for(blocked_range<size_t>(0,n_ref),
        [=] (const blocked_range<size_t>& br)
            {
            float dx_inv = 1.0f / m_dx;
            float dy_inv = 1.0f / m_dy;
            float rmaxsq = m_r_max * m_r_max;
            // indexer for bond list
            Index2D a_i = Index2D(m_n_bonds, n_ref);
            // indexer for bond map
            Index2D b_i = Index2D(m_nbins_x, m_nbins_y);

            for(size_t i=br.begin(); i!=br.end(); ++i)
                {
                // create vector of pjdx to verify
                std::vector<unsigned int> current_pjdxs;
                current_pjdxs.resize(0);
                std::vector<unsigned int> touched_pjdxs;
                touched_pjdxs.resize(0);
                // iterate through current bonds and put into current bond array
                unsigned int num_bonds = (unsigned int) m_bond_tracker_array[i].size();
                // safer to iterate through
                for (unsigned int b_idx = 0; b_idx < num_bonds; b_idx++)
                    {
                    unsigned int l_pjdx = m_bond_tracker_array[i][b_idx].first;
                    current_pjdxs.push_back(l_pjdx);
                    }
                // std::vector<unsigned int>::iterator print_iterator;
                // for (print_iterator = current_pjdxs.begin(); print_iterator != current_pjdxs.end(); ++print_iterator)
                //     {
                //     printf("%u ", (*print_iterator));
                //     }

                // get position, orientation of particle i
                vec3<float> ref_pos = ref_points[i];
                float ref_angle = ref_orientations[i];
                // get cell for particle i
                unsigned int ref_cell = m_lc->getCell(ref_pos);

                //loop over neighbor cells
                const std::vector<unsigned int>& neigh_cells = m_lc->getCellNeighbors(ref_cell);
                for (unsigned int neigh_idx = 0; neigh_idx < neigh_cells.size(); neigh_idx++)
                    {
                    // get neighbor cell
                    unsigned int neigh_cell = neigh_cells[neigh_idx];

                    // iterate over the particles in that cell
                    locality::LinkCell::iteratorcell it = m_lc->itercell(neigh_cell);
                    for (unsigned int j = it.next(); !it.atEnd(); j=it.next())
                        {
                        //compute r between the two particles
                        vec3<float> delta = m_box.wrap(points[j] - ref_pos);

                        float rsq = dot(delta, delta);
                        // particle cannot pair with itself...i != j is probably better?
                        // if ((rsq < 1e-6) || (rsq > rmaxsq))
                        if ((i == j) || (rsq > rmaxsq))
                            // this should skip i == j and if rsq would segfault in the bond map lookup
                            // and the set difference should catch the b2u
                            {
                            continue;
                            }
                        std::vector<unsigned int>::iterator insert_idx = std::upper_bound(touched_pjdxs.begin(), touched_pjdxs.end(), j);
                        touched_pjdxs.insert(insert_idx, j);
                        // determine which histogram bin to look in
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
                        unsigned int ibin_x = _mm_cvtt_ss2si(_mm_load_ss(&binx));
                        unsigned int ibin_y = _mm_cvtt_ss2si(_mm_load_ss(&biny));
                        #else
                        unsigned int ibin_x = (unsigned int)(binx);
                        unsigned int ibin_y = (unsigned int)(biny);
                        #endif

                        // find the bond
                        // bin if bond is tracked
                        bool isBondTracked = false;
                        unsigned int bond;
                        // printf("accessing bond map\n");
                        if ((ibin_x < m_nbins_x) && (ibin_y < m_nbins_y))
                            {
                            // find the bond that corresponds to this point
                            bond = m_bond_map[b_i(ibin_x, ibin_y)];
                            // get the index from the map
                            auto list_idx = m_list_map.find(bond);
                            if (list_idx != m_list_map.end())
                                {
                                isBondTracked = true;
                                }
                            }
                        // log the bond
                        // check if pjdx in the current bond list
                        std::vector<unsigned int>::iterator it_pjdx;
                        it_pjdx = std::find_if(current_pjdxs.begin(), current_pjdxs.end(), FindParticle(j));
                        if (it_pjdx != current_pjdxs.end())
                            {
                            // found pjdx; algorithm shouldn't double-add
                            // get index into the array
                            auto l_index = std::distance(current_pjdxs.begin(), it_pjdx);
                            // get current value of bond
                            std::vector<unsigned int> l_bond_values = m_bond_tracker_array[i][l_index].second;
                            unsigned int l_bond_label = l_bond_values[0];
                            if (isBondTracked)
                                {
                                if (l_bond_label == bond)
                                    {
                                    // bound to bound transition; increment bond lifetime and overall lifetime
                                    m_bond_tracker_array[i][l_index].second[1]++;
                                    m_bond_tracker_array[i][l_index].second[2]++;
                                    }
                                else
                                    {
                                    // they don't match. bound to bound transition
                                    // get the bond lifetime
                                    unsigned int b_lifetime = l_bond_values[1];
                                    m_bond_lifetime_array.push_back(b_lifetime);
                                    // update the bond
                                    m_bond_tracker_array[i][l_index].second[0] = bond;
                                    // update the bond lifetime count to 1
                                    m_bond_tracker_array[i][l_index].second[1] = 1;
                                    // increment the overall lifetime
                                    m_bond_tracker_array[i][l_index].second[2]++;
                                    }
                                }
                            else
                                {
                                // bond isn't tracked; bound to unbound transition
                                unsigned int b_lifetime = l_bond_values[1];
                                m_bond_lifetime_array.push_back(b_lifetime);
                                unsigned int o_lifetime = l_bond_values[2];
                                m_overall_lifetime_array.push_back(o_lifetime);
                                // delete the pair from the array
                                m_bond_tracker_array[i].erase(m_bond_tracker_array[i].begin()+l_index);
                                // do I need to put this in here?
                                current_pjdxs.erase(current_pjdxs.begin()+l_index);
                                }
                            }
                        else
                            {
                            if (isBondTracked)
                                {
                                // didn't find, need to add
                                std::pair< unsigned int, std::vector<unsigned int> > new_pjdx;
                                // new_pjdx.resize(0);
                                std::vector<unsigned int> new_element;
                                new_element.resize(0);
                                // bond label
                                new_element.push_back(bond);
                                // bond lifetime count
                                new_element.push_back(1);
                                // overall lifetime count
                                new_element.push_back(1);
                                new_pjdx.first = j;
                                new_pjdx.second = new_element;
                                // do for current array first
                                std::vector< unsigned int >::iterator insert_current_pjdx;
                                insert_current_pjdx = std::upper_bound(current_pjdxs.begin(), current_pjdxs.end(), j);
                                // find the distance
                                auto l_index = std::distance(current_pjdxs.begin(), insert_current_pjdx);
                                current_pjdxs.insert(insert_current_pjdx, j);
                                // should already be initialized
                                // m_bond_tracker_array[i].push_back(new_pjdx);
                                m_bond_tracker_array[i].insert(m_bond_tracker_array[i].begin()+l_index, new_pjdx);
                                }
                            }
                        }
                    }
                // now compare pjdx we've check vs. the ones we started with
                std::vector<unsigned int> b2u;
                b2u.resize(0);
                std::set_difference(current_pjdxs.begin(), current_pjdxs.end(), touched_pjdxs.begin(),
                    touched_pjdxs.end(), std::back_inserter(b2u));
                // std::vector<unsigned int>::iterator print_iterator;
                // looks like maybe there's uninitialized memory?
                if (b2u.size() > 0)
                // if (false)
                    {
                    // printf("current_pjdxs:\n");
                    // printf("%u: ", i);
                    // for (print_iterator = current_pjdxs.begin(); print_iterator != current_pjdxs.end(); ++print_iterator)
                    //     {
                    //     printf("%u ", (*print_iterator));
                    //     }
                    // printf("\n");
                    // printf("touched_pjdxs:\n");
                    // printf("%u: ", i);
                    // for (print_iterator = touched_pjdxs.begin(); print_iterator != touched_pjdxs.end(); ++print_iterator)
                    //     {
                    //     printf("%u ", (*print_iterator));
                    //     }
                    // printf("\n");
                    // printf("set difference:\n");
                    // printf("%u: ", i);
                    // for (print_iterator = b2u.begin(); print_iterator != b2u.end(); ++print_iterator)
                    //     {
                    //     printf("%u ", (*print_iterator));
                    //     }
                    // printf("b2u?\n");
                    for (std::vector<unsigned int>::iterator j = b2u.begin(); j != b2u.end(); ++j)
                        {
                        std::vector<unsigned int>::iterator it_pjdx;
                        // I think this should be ok
                        it_pjdx = std::find_if(current_pjdxs.begin(),
                            current_pjdxs.end(), FindParticle(*j));
                        if (it_pjdx != current_pjdxs.end())
                            {
                            // get the index
                            auto l_index = std::distance(current_pjdxs.begin(), it_pjdx);
                            // get the values
                            std::vector<unsigned int> l_bond_values = m_bond_tracker_array[i][l_index].second;
                            // put values into the arrays
                            unsigned int b_lifetime = l_bond_values[1];
                            m_bond_lifetime_array.push_back(b_lifetime);
                            unsigned int o_lifetime = l_bond_values[2];
                            m_overall_lifetime_array.push_back(o_lifetime);
                            // delete the pair from the array
                            // printf("time to erase\n");
                            std::vector<std::pair< unsigned int, std::vector<unsigned int> > >::iterator print_iterator;
                            // printf("erasing b2u:\n");
                            // printf("%u: ", i);
                            // for (print_iterator = m_bond_tracker_array[i].begin(); print_iterator != m_bond_tracker_array[i].end(); ++print_iterator)
                                // {
                                // printf("%u ", print_iterator->first);
                                // }
                            // printf("\n");
                            m_bond_tracker_array[i].erase(m_bond_tracker_array[i].begin()+l_index);
                            current_pjdxs.erase(current_pjdxs.begin()+l_index);
                            // printf("%u: ", i);
                            // for (print_iterator = m_bond_tracker_array[i].begin(); print_iterator != m_bond_tracker_array[i].end(); ++print_iterator)
                                // {
                                // printf("%u ", print_iterator->first);
                                // }
                            // printf("\n");
                            // printf("erased\n");
                            }
                        else
                            {
                            // this line should NEVER BE REACHED
                            printf("this line should never be reached!\n");
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


