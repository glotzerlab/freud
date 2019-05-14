// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef MATCH_ENV_H
#define MATCH_ENV_H

#include <algorithm>
#include <complex>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

#include "Box.h"
#include "VectorMath.h"
#include "Cluster.h"
#include "NearestNeighbors.h"
#include "BiMap.h"
#include "brute_force.h"

/*! \file MatchEnv.h
    \brief Particle environment matching
*/

namespace freud { namespace environment {
//! Clusters particles according to whether their local environments match or not, according to various shape matching metrics

//! My environment data structure
struct Environment
    {
    //! Constructor.
    Environment() : vecs(0), vec_ind(0)
        {
        env_ind = 0;
        num_vecs = 0;
        ghost = false;
        proper_rot = rotmat3<float>(); // the default construction is the identity matrix
        }
    //! Add a vector to define the local environment
    void addVec(vec3<float> vec)
        {
        vecs.push_back(vec);
        vec_ind.push_back(num_vecs);
        num_vecs++;
        }

    unsigned int env_ind;            //!< The index of the environment
    std::vector<vec3<float> > vecs;  //!< The vectors that define the environment
    //! Is this environment a ghost? Do we ignore it when we compute actual
    //  physical quantities associated with all environments?
    bool ghost;
    unsigned int num_vecs;           //!< The number of vectors defining the environment currently
    //! The order that the vectors must be in to define the environment
    std::vector<unsigned int> vec_ind;
    //! The rotation that defines the proper orientation of the environment
    rotmat3<float> proper_rot;
    };

//! General disjoint set class, taken mostly from Cluster.h
class EnvDisjointSet
    {
    public:
        //! Constructor
        EnvDisjointSet(unsigned int Np);
        //! Merge two sets
        void merge(const unsigned int a, const unsigned int b, BiMap<unsigned int, unsigned int> vec_map, rotmat3<float> rotation);
        //! Find the set with a given element
        unsigned int find(const unsigned int c);
        //! Return ALL nodes in the tree that correspond to the head index m
        std::vector<unsigned int> findSet(const unsigned int m);
        //! Get the vectors corresponding to environment head index m. Vectors are averaged over all members of the environment cluster.
        std::shared_ptr<vec3<float> > getAvgEnv(const unsigned int m);
        //! Get the vectors corresponding to index m in the dj set
        std::vector<vec3<float> > getIndividualEnv(const unsigned int m);

        std::vector<Environment> s;      //!< The disjoint set data
        std::vector<unsigned int> rank;  //!< The rank of each tree in the set
        unsigned int m_max_num_neigh;    //!< The maximum number of neighbors in any environment in the set
    };

class MatchEnv
    {
    public:
        //! Constructor
        /**Constructor for Match-Environment analysis class.  After creation, call cluster to agnostically calculate clusters grouped by matching environment,
        or matchMotif to match all particle environments against an input motif.  Use accessor functions to retrieve data.
        @param rmax Cutoff radius for cell list and clustering algorithm.  Values near first minimum of the rdf are recommended.
        @param k Number of nearest neighbors taken to define the local environment of any given particle.
        **/
        MatchEnv(const box::Box& box, float rmax, unsigned int k=12);

        //! Destructor
        ~MatchEnv();

        //! Construct and return a local environment surrounding the particle indexed by i. Set the environment index to env_ind.
        //! if hard_r is true, add all particles that fall within the threshold of m_rmaxsq to the environment
        Environment buildEnv(const size_t *neighbor_list, size_t num_bonds,
                             size_t &bond, const vec3<float> *points,
                             unsigned int i, unsigned int env_ind, bool hard_r);

        //! Determine clusters of particles with matching environments
        //! env_nlist is the neighborlist used to build the environment of every particle.
        //! nlist is the neighborlist used to determine the neighbors against which to compare environments for every particle, if hard_r = False.
        //! The threshold is a unitless number, which we multiply by the length scale of the MatchEnv instance, rmax.
        //! This quantity is the maximum squared magnitude of the vector difference between two vectors, below which you call them matching.
        //! Note that ONLY values of (threshold < 2) make any sense, since 2*rmax is the absolute maximum difference between any two environment vectors.
        //! If hard_r is true, add all particles that fall within the threshold of m_rmaxsq to the environment
        //! The bool registration controls whether we first use brute force registration to orient the second set of vectors such that it minimizes the RMSD between the two sets
        //! If global is true, do an exhaustive search wherein you compare the environments of every single pair of particles in the simulation. If global is false, only compare the environments of neighboring particles.
        void cluster(const freud::locality::NeighborList *env_nlist, const freud::locality::NeighborList *nlist, const vec3<float> *points, unsigned int Np, float threshold, bool hard_r=false, bool registration=false, bool global=false);

        //! Determine whether particles match a given input motif, characterized by refPoints (of which there are numRef)
        //! The threshold is a unitless number, which we multiply by the length scale of the MatchEnv instance, rmax.
        //! This quantity is the maximum squared magnitude of the vector difference between two vectors, below which you call them matching.
        //! Note that ONLY values of (threshold < 2) make any sense, since 2*rmax is the absolute maximum difference between any two environment vectors.
        //! The bool registration controls whether we first use brute force registration to orient the second set of vectors such that it minimizes the RMSD between the two sets
        void matchMotif(const freud::locality::NeighborList *nlist, const vec3<float> *points, unsigned int Np, const vec3<float> *refPoints, unsigned int numRef, float threshold, bool registration=false);

        //! Rotate (if registration=True) and permute the environments of all particles to minimize their RMSD wrt a given input motif, characterized by refPoints (of which there are numRef).
        //! Returns a vector of minimal RMSD values, one value per particle.
        //! NOTE that this does not guarantee an absolutely minimal RMSD. It doesn't figure out the optimal permutation
        //! of BOTH sets of vectors to minimize the RMSD. Rather, it just figures out the optimal permutation of the second set, the vector set used in the argument below.
        //! To fully solve this, we need to use the Hungarian algorithm or some other way of solving the so-called assignment problem.
        std::vector<float> minRMSDMotif(const freud::locality::NeighborList *nlist, const vec3<float> *points, unsigned int Np, const vec3<float> *refPoints, unsigned int numRef, bool registration=false);

        //! Renumber the clusters in the disjoint set dj from zero to num_clusters-1
        void populateEnv(EnvDisjointSet dj, bool reLabel=true);

        //! Is the (PROPERLY REGISTERED) environment e1 similar to the (PROPERLY REGISTERED) environment e2?
        //! If so, return a std::pair of the rotation matrix that takes the vectors of e2 to the vectors of e1 AND the mapping between the properly indexed vectors of the environments that will make them correspond to each other.
        //! If not, return a std::pair of the identity matrix AND an empty map.
        //! The threshold is a unitless number, which we multiply by the length scale of the MatchEnv instance, rmax.
        //! This quantity is the maximum squared magnitude of the vector difference between two vectors, below which you call them matching.
        //! The bool registration controls whether we first use brute force registration to orient the second set of vectors such that it minimizes the RMSD between the two sets
        std::pair<rotmat3<float>, BiMap<unsigned int, unsigned int> > isSimilar(Environment& e1, Environment& e2, float threshold_sq, bool registration);

        //! Overload: is the set of vectors refPoints1 similar to the set of vectors refPoints2?
        //! Construct the environments accordingly, and utilize isSimilar() as above.
        //! Return a std map for ease of use.
        //! The bool registration controls whether we first use brute force registration to orient the second set of vectors such that it minimizes the RMSD between the two sets.
        //! If registration=True, then refPoints2 is CHANGED by this function.
        std::map<unsigned int, unsigned int> isSimilar(const vec3<float> *refPoints1, vec3<float> *refPoints2, unsigned int numRef, float threshold_sq, bool registration);

        // Get the somewhat-optimal RMSD between the (PROPERLY REGISTERED) environment e1 and the (PROPERLY REGISTERED) environment e2.
        // Return a std::pair of the rotation matrix that takes the vectors of e2 to the vectors of e1 AND the mapping between the properly indexed vectors of the environments that gives this RMSD.
        // Populate the associated minimum RMSD.
        // The bool registration controls whether we first use brute force registration to orient the second set of vectors such that it minimizes the RMSD between the two sets.
        // NOTE that this does not guarantee an absolutely minimal RMSD. It doesn't figure out the optimal permutation
        // of BOTH sets of vectors to minimize the RMSD. Rather, it just figures out the optimal permutation of the second set, the vector set used in the argument below.
        // To fully solve this, we need to use the Hungarian algorithm or some other way of solving the so-called assignment problem.
        std::pair<rotmat3<float>, BiMap<unsigned int, unsigned int> > minimizeRMSD(Environment& e1, Environment& e2, float& min_rmsd, bool registration);

        // Overload: Get the somewhat-optimal RMSD between the set of vectors refPoints1 and the set of vectors refPoints2.
        // Construct the environments accordingly, and utilize minimizeRMSD() as above.
        // Arguments are pointers to interface directly with python.
        // Return a std::map (for ease of use) with the mapping between vectors refPoints1 and refPoints2 that gives this RMSD.
        // Populate the associated minimum RMSD.
        // The bool registration controls whether we first use brute force registration to orient the second set of vectors such that it minimizes the RMSD between the two sets.
        // NOTE that this does not guarantee an absolutely minimal RMSD. It doesn't figure out the optimal permutation
        // of BOTH sets of vectors to minimize the RMSD. Rather, it just figures out the optimal permutation of the second set, the vector set used in the argument below.
        // To fully solve this, we need to use the Hungarian algorithm or some other way of solving the so-called assignment problem.
        std::map<unsigned int, unsigned int> minimizeRMSD(const vec3<float> *refPoints1, vec3<float> *refPoints2, unsigned int numRef, float& min_rmsd, bool registration);

        //! Get a reference to the particles, indexed into clusters according to their matching local environments
        std::shared_ptr<unsigned int> getClusters()
            {
            return m_env_index;
            }

        //! Reset the simulation box
        void setBox(const box::Box newbox)
            {
            m_box = newbox;
            }

        //! Returns the set of vectors defining the environment indexed by i (indices culled from m_env_index)
        std::shared_ptr< vec3<float> > getEnvironment(unsigned int i)
            {
            std::map<unsigned int, std::shared_ptr<vec3<float> > >::iterator it = m_env.find(i);
            std::shared_ptr<vec3<float> > vecs = it->second;
            return vecs;
            }

        //! Returns the entire m_Np by m_k by 3 matrix of all environments for all particles
        std::shared_ptr<vec3<float> > getTotEnvironment()
            {
            return m_tot_env;
            }

        unsigned int getNP()
            {
            return m_Np;
            }

        unsigned int getNumClusters()
            {
            return m_num_clusters;
            }
        unsigned int getNumNeighbors()
            {
            return m_k;
            }
        unsigned int getMaxNumNeighbors()
            {
            return m_maxk;
            }

    private:
        box::Box m_box;              //!< Simulation box
        float m_rmax;                       //!< Maximum cutoff radius at which to determine local environment
        float m_rmaxsq;                     //!< Square of m_rmax
        float m_k;                          //!< Default number of nearest neighbors used to determine which environments are compared during local environment clustering.
                                            //!< If hard_r=false, this is also the number of neighbors in each local environment.
        unsigned int m_maxk;                //!< Maximum number of neighbors in any particle's local environment. If hard_r=false, m_maxk = m_k.
                                            //!< In the cluster method it is also possible to provide two separate neighborlists, one for environments and one for clustering.
        unsigned int m_Np;                  //!< Last number of points computed
        unsigned int m_num_clusters;        //!< Last number of local environments computed

        std::shared_ptr<unsigned int> m_env_index;                              //!< Cluster index determined for each particle
        std::map<unsigned int, std::shared_ptr<vec3<float> > > m_env;           //!< Dictionary of (cluster id, vectors) pairs
        std::shared_ptr<vec3<float> > m_tot_env;                                //!< m_NP by m_maxk by 3 matrix of all environments for all particles
    };

}; }; // end namespace freud::environment

#endif // MATCH_ENV_H
