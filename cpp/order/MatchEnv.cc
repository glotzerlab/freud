#include "MatchEnv.h"
#include "Cluster.h"
#include <map>
//#include <boost/math/special_functions.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>

using namespace std;

namespace freud { namespace order {

MatchEnv::MatchEnv(float rmax)
    :m_rmax(rmax)
    {
    m_Np = 0;
    if (m_rmax < 0.0f)
        throw invalid_argument("rmax must be positive");
    }

// Construct and return a local environment surrounding a particle
Environment MatchEnv::buildEnv(const vec3<float> *points, unsigned int i)
    {
    Environment ei = Environment(i);

    // get the cell the point is in
    vec3<float> p = points[i];
    unsigned int cell = m_lc.getCell(p);

    // loop over all neighboring cells
    const std::vector<unsigned int>& neigh_cells = m_lc.getCellNeighbors(cell);
    for (unsigned int neigh_idx = 0; neigh_idx < neigh_cells.size(); neigh_idx++)
        {
        unsigned int neigh_cell = neigh_cells[neigh_idx];

        // iterate over the particles in that cell
        locality::LinkCell::iteratorcell it = m_lc.itercell(neigh_cell);
        for (unsigned int j = it.next(); !it.atEnd(); j=it.next())
            {
            // compute r between the two particles
            vec3<float> delta = m_box.wrap(p - points[j]);
            float rsq = dot(delta, delta);

            if (rsq < rmaxsq)
                {
                ei.addVec(delta);
                }
            }
        }
    return ei;
    }

// Determine clusters of particles with matching environments
void MatchEnv::compute(const vec3<float> *points, const trajectory::Box& box, unsigned int Np)
    {
    m_Np = Np;
    m_box = box;
    m_lc.computeCellList(m_box,points,Np);
    float rmaxsq = m_rmax * m_rmax;

    // loop through points
    for (unsigned int i = 0; i < Np; i++)
        {
        // 1. make an Environment instance and add it to the vector m_env
        Environment ei = buildEnv(i);
        m_env.push_back(ei);

        // 2. loop over the neighbors again. Now, construct the environment for the neighboring particle and compare
        // get the cell the point is in
        vec3<float> p = points[i];
        unsigned int cell = m_lc.getCell(p);

        // loop over all neighboring cells
        const std::vector<unsigned int>& neigh_cells = m_lc.getCellNeighbors(cell);
        for (unsigned int neigh_idx = 0; neigh_idx < neigh_cells.size(); neigh_idx++)
            {
            unsigned int neigh_cell = neigh_cells[neigh_idx];

            // iterate over the particles in that cell
            locality::LinkCell::iteratorcell it = m_lc.itercell(neigh_cell);
            for (unsigned int j = it.next(); !it.atEnd(); j=it.next())
                {
                // only construct the environment and do all this rigamarole if we haven't looked at this (i,j) combo before
                // if (i < j)
                int blah=0;
                }
            }
        }
    }

    //
    // freud::cluster::DisjointSet dj(Np);
    //
    //             // loop over particles i, then loop over their neighbors j
    //             if (i != j)
    //                 {
    //                 // if we belong in the same cluster
    //                     {
    //                         // merge the two sets using the disjoint set
    //                         uint32_t a = dj.find(i);
    //                         uint32_t b = dj.find(j);
    //                         if (a != b)
    //                             dj.merge(a,b);
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }
    //
    // // done looping over points. All clusters are now determined. Renumber them from zero to num_clusters-1.
    // map<uint32_t, uint32_t> label_map;
    //
    // // go over every point
    // uint32_t cur_set = 0;
    // for (uint32_t i = 0; i < Np; i++)
    //     {
    //     uint32_t s = dj.find(i);
    //
    //     // insert it into the mapping if we haven't seen this one yet
    //     if (label_map.count(s) == 0)
    //         {
    //         label_map[s] = cur_set;
    //         cur_set++;
    //         }
    //
    //     // label this point in cluster_idx
    //     m_cluster_idx[i] = label_map[s];
    //     }
    //
    // // cur_set is now the number of clusters
    // m_num_clusters = cur_set;
    // }

}; };// end namespace freud::match_env;
