#include <algorithm>
#include <stdexcept>
#include <complex>
#include <utility>
#include <vector>
#include <tbb/tbb.h>
#include <boost/math/special_functions/spherical_harmonic.hpp>

#include "NearestNeighbors.h"
#include "ScopedGILRelease.h"
#include "HOOMDMatrix.h"

using namespace std;
using namespace tbb;

/*! \file NearestNeighbors.h
  \brief Compute the hexatic order parameter for each particle
*/

namespace freud { namespace locality {

// stop using
NearestNeighbors::NearestNeighbors():
    m_box(box::Box()), m_rmax(0), m_num_neighbors(0), m_scale(0), m_strict_cut(false), m_num_points(0), m_num_ref(0),
    m_deficits()
    {
    m_lc = new locality::LinkCell();
    m_deficits = 0;
    }

NearestNeighbors::NearestNeighbors(float rmax,
                                   unsigned int num_neighbors,
                                   float scale,
                                   bool strict_cut):
    m_box(box::Box()), m_rmax(rmax), m_num_neighbors(num_neighbors), m_scale(scale), m_strict_cut(strict_cut), m_num_points(0),
    m_num_ref(0), m_deficits()
    {
    m_lc = new locality::LinkCell(m_box, m_rmax);
    m_deficits = 0;
    }

NearestNeighbors::~NearestNeighbors()
    {
    delete m_lc;
    }

//! Utility function to sort a pair<float, unsigned int> on the first
//! element of the pair
bool compareRsqVectors(const pair<float, unsigned int> &left,
                       const pair<float, unsigned int> &right)
    {
    return left.first < right.first;
    }

void NearestNeighbors::setCutMode(const bool strict_cut)
    {
    m_strict_cut = strict_cut;
    }

void NearestNeighbors::compute(const box::Box& box,
                               const vec3<float> *ref_pos,
                               unsigned int num_ref,
                               const vec3<float> *pos,
                               unsigned int num_points)
    {
    m_box = box;
    // reallocate the output array if it is not the right size
    if (num_ref != m_num_ref)
        {
        m_rsq_array = std::shared_ptr<float>(new float[num_ref * m_num_neighbors], std::default_delete<float[]>());
        m_neighbor_array = std::shared_ptr<unsigned int>(new unsigned int[num_ref * m_num_neighbors], std::default_delete<unsigned int[]>());
        }
    // fill with padded values; rsq set to -1, neighbors set to UINT_MAX
    memset((void*)m_rsq_array.get(), -1, sizeof(float)*num_ref*m_num_neighbors);
    memset((void*)m_neighbor_array.get(), UINT_MAX, sizeof(float)*num_ref*m_num_neighbors);
    // find the nearest neighbors
    do
        {
        // compute the cell list
        m_lc->computeCellList(m_box, pos, num_points);

        m_deficits = 0;
        parallel_for(blocked_range<size_t>(0,num_ref),
          [=] (const blocked_range<size_t>& r)
          {
          float rmaxsq = m_rmax * m_rmax;
          // tuple<> is c++11, so for now just make a pair with pairs inside
          // this data structure holds rsq, idx
          vector< pair<float, unsigned int> > neighbors;
          Index2D b_i = Index2D(m_num_neighbors, num_ref);
          for(size_t i=r.begin(); i!=r.end(); ++i)
              {
              // If we have found an incomplete set of neighbors, end now and rebuild
              if(m_deficits > 0)
                  break;
              neighbors.clear();

              //get cell point is in
              vec3<float> posi = ref_pos[i];
              unsigned int ref_cell = m_lc->getCell(posi);
              unsigned int num_adjacent = 0;

              //loop over neighboring cells
              const std::vector<unsigned int>& neigh_cells = m_lc->getCellNeighbors(ref_cell);
              for (unsigned int neigh_idx = 0; neigh_idx < neigh_cells.size(); neigh_idx++)
                  {
                  unsigned int neigh_cell = neigh_cells[neigh_idx];

                  //iterate over particles in cell
                  locality::LinkCell::iteratorcell it = m_lc->itercell(neigh_cell);
                  for (unsigned int j = it.next(); !it.atEnd(); j = it.next())
                      {

                      //compute r between the two particles
                      vec3<float>rij = m_box.wrap(pos[j] - posi);
                      const float rsq = dot(rij, rij);

                      // adds all neighbors within rsq to list of possible neighbors
                      if ((rsq < rmaxsq) && (i != j))
                          {
                          neighbors.push_back(pair<float, unsigned int>(rsq, j));
                          num_adjacent++;
                          }
                      }
                  }

              // Add to the deficit count if necessary
              if(num_adjacent < m_num_neighbors)
                  m_deficits += (m_num_neighbors - num_adjacent);
              else
                  {
                  // sort based on rsq
                  sort(neighbors.begin(), neighbors.end(), compareRsqVectors);
                  for (unsigned int k = 0; k < m_num_neighbors; k++)
                      {
                      // put the idx into the neighbor array
                      m_rsq_array.get()[b_i(k, i)] = neighbors[k].first;
                      m_neighbor_array.get()[b_i(k, i)] = neighbors[k].second;
                      }
                  }
              }
          });

        // Increase m_rmax
        if((m_deficits > 0) && !(m_strict_cut))
            {
            m_rmax *= m_scale;
            // check if new r_max would be too large for the cell width
            vec3<float> L = m_box.getNearestPlaneDistance();
            bool too_wide =  m_rmax > L.x/2.0 || m_rmax > L.y/2.0;
            if (!m_box.is2D())
                {
                too_wide |=  m_rmax > L.z/2.0;
                }
            if (too_wide)
                {
                // throw runtime_warning("r_max has become too large to create a viable cell.");
                // for now print
                printf("r_max has become too large to create a viable cell. Returning neighbors found\n");
                m_deficits = 0;
                break;
                }
            m_lc->setCellWidth(m_rmax);
            }
        } while((m_deficits > 0) && !(m_strict_cut));
    // save the last computed number of particles
    m_num_ref = num_ref;
    m_num_points = num_points;
    }

}; }; // end namespace freud::locality
