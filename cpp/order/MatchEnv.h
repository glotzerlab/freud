#ifndef _MATCH_ENV_H__
#define _MATCH_ENV_H__

#include <boost/shared_array.hpp>
#include <boost/bimap.hpp>

#include "HOOMDMath.h"
#include "VectorMath.h"

#include <vector>
#include <set>

#include "Cluster.h"
#include "NearestNeighbors.h"

#include "trajectory.h"
#include <stdexcept>
#include <complex>
#include <map>
#include <algorithm>
#include <iostream>


namespace freud { namespace order {
//! Clusters particles according to whether their local environments match or not, according to various shape matching metrics

//! My environment data structure
struct Environment
    {
    //! Constructor.
    Environment(unsigned int n) : vecs(0), vec_ind(0)
        {
        num_neigh = n;
        env_ind = 0;
        num_vecs = 0;
        }
    //! Assimilate the set of vectors v2 (INDEXED PROPERLY) into this environment
    void assimilate(std::vector< vec3<float> > v2)
        {
        int blah=0;
        }
    //! Add a vector to define the local environment
    void addVec(vec3<float> vec)
        {
        if (num_vecs > num_neigh)
            {
            fprintf(stderr, "Current number of vecs is %d\n", num_vecs);
            throw std::invalid_argument("You've added too many vectors to the environment!");
            }
        vecs.push_back(vec);
        vec_ind.push_back(num_vecs);
        num_vecs++;
        }

    unsigned int env_ind;                   //!< The index of the environment
    std::vector<vec3<float> > vecs;         //!< The vectors that define the environment
    unsigned int num_vecs;                  //!< The number of vectors defining the environment currently
    unsigned int num_neigh;                 //!< The maximum allowed number of vectors to define the environment
    std::vector<unsigned int> vec_ind;      //!< The order that the vectors must be in to define the environment
    };

//! General disjoint set class, taken mostly from Cluster.h
class EnvDisjointSet
    {
    public:
        //! Constructor
        EnvDisjointSet(unsigned int num_neigh, unsigned int Np);
        //! Merge two sets
        void merge(const unsigned int a, const unsigned int b, boost::bimap<unsigned int, unsigned int> vec_map);
        //! Find the set with a given element
        unsigned int find(const unsigned int c);
        std::vector<Environment> s;         //!< The disjoint set data
        std::vector<unsigned int> rank;     //!< The rank of each tree in the set
        unsigned int m_num_neigh;           //!< The number of neighbors allowed per environment
    };

class MatchEnv
    {
    public:
        //! Constructor
        /**Constructor for Match-Environment analysis class.  After creation, call compute to calculate clusters grouped by matching environment.  Use accessor functions to retrieve data.
        @param rmax Cutoff radius for cell list and clustering algorithm.  Values near first minimum of the rdf are recommended.
        **/
        MatchEnv(const trajectory::Box& box, float rmax, unsigned int k=12);

        //! Destructor
        ~MatchEnv();

        //! Construct and return a local environment surrounding a particle indexed by i
        Environment buildEnv(const vec3<float> *points, unsigned int i);

        //! Determine clusters of particles with matching environments
        void compute(const vec3<float> *points, unsigned int Np, float threshold);

        //! Is the environment e1 similar to the environment e2?
        //! If so, return the mapping between the vectors of the environments that will make them correspond to each other.
        //! If not, return an empty map
        boost::bimap<unsigned int, unsigned int> isSimilar(Environment e1, Environment e2, float threshold_sq);

        //! Get a reference to the particles, indexed into clusters according to their matching local environments
        boost::shared_array<unsigned int> getClusters()
            {
            return m_env_index;
            }

        //! Reset the simulation box
        void setBox(const trajectory::Box newbox)
            {
            m_box = newbox;
            delete m_nn;
            m_nn = new locality::NearestNeighbors(m_rmax, m_k);
            }


        //! Returns the set of vectors defining the environment indexed by i (indices culled from m_env_index)
        std::vector< vec3<float> > getEnvironment(unsigned int i)
            {
            std::map<unsigned int, std::vector<vec3<float> > >::const_iterator it = vec_map.left.find(a_ind);
            std::vector<vec3<float> > vecs = m_env[i];
            return vecs;
            }

        unsigned int getNP()
            {
            return m_Np;
            }

    private:
        trajectory::Box m_box;              //!< Simulation box
        float m_rmax;                       //!< Maximum cutoff radius at which to determine local environment
        float m_rmaxsq;                     //!< square of m_rmax
        float m_k;                          //!< Number of nearest neighbors used to determine local environment
        locality::NearestNeighbors *m_nn;   //!< NearestNeighbors to bin particles for the computation of local environments
        unsigned int m_Np;                  //!< Last number of points computed

        boost::shared_array<unsigned int> m_env_index;              //!< Cluster index determined for each particle
        std::map<unsigned int, std::vector<vec3<float> > > m_env;   //!< Dictionary of (cluster id, vectors) pairs
    };

}; }; // end namespace freud::match_env

#endif // _MATCH_ENV_H__
