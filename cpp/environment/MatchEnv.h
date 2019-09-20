// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef MATCH_ENV_H
#define MATCH_ENV_H

#include <algorithm>
#include <map>
#include <stdexcept>
#include <vector>

#include "BiMap.h"
#include "Box.h"
#include "ManagedArray.h"
#include "NeighborList.h"
#include "VectorMath.h"
#include "brute_force.h"

/*! \file MatchEnv.h
    \brief Particle environment matching
*/

namespace freud { namespace environment {
//! Clusters particles according to whether their local environments match or not, according to various shape
//! matching metrics

//! My environment data structure
struct Environment
{
    //! Constructor.
    Environment(bool ghost=false) : env_ind(0), vecs(0), ghost(ghost), num_vecs(0), vec_ind(0), proper_rot() {}

    //! Add a vector to define the local environment
    void addVec(vec3<float> vec)
    {
        vecs.push_back(vec);
        vec_ind.push_back(num_vecs);
        num_vecs++;
    }

    unsigned int env_ind;          //!< The index of the environment
    std::vector<vec3<float>> vecs; //!< The vectors that define the environment
    //! Is this environment a ghost? Do we ignore it when we compute actual
    //  physical quantities associated with all environments?
    bool ghost;
    unsigned int num_vecs; //!< The number of vectors defining the environment currently
    //! The order that the vectors must be in to define the environment
    std::vector<unsigned int> vec_ind;
    //! The rotation that defines the proper orientation of the environment
    rotmat3<float> proper_rot;
};

//! General disjoint set class, taken mostly from Cluster.h
struct EnvDisjointSet
{
    //! Constructor
    EnvDisjointSet(unsigned int Np);
    //! Merge two sets
    void merge(const unsigned int a, const unsigned int b, BiMap<unsigned int, unsigned int> vec_map,
               rotmat3<float> rotation);
    //! Find the set with a given element
    unsigned int find(const unsigned int c);
    //! Return ALL nodes in the tree that correspond to the head index m
    std::vector<unsigned int> findSet(const unsigned int m);
    //! Get the vectors corresponding to environment head index m. Vectors are averaged over all members of
    //! the environment cluster.
    std::vector<vec3<float> > getAvgEnv(const unsigned int m);
    //! Get the vectors corresponding to index m in the dj set
    std::vector<vec3<float>> getIndividualEnv(const unsigned int m);

    std::vector<Environment> s;     //!< The disjoint set data
    std::vector<unsigned int> rank; //!< The rank of each tree in the set
    unsigned int m_max_num_neigh;   //!< The maximum number of neighbors in any environment in the set
};

class MatchEnv
{
public:
    //! Constructor
    /*!
     * Constructor for Match-Environment analysis class.  After creation, call
     * cluster to agnostically calculate clusters grouped by matching
     * environment, or matchMotif to match all particle environments against an
     * input motif. Use accessor functions to retrieve data. 
     *
     * \param box The system box.
     * \param r_max Cutoff radius for cell list and clustering algorithm. Values near first minimum of the rdf are recommended.  
     * \param num_neighbors Number of nearest neighbors taken to define the local environment of any given particle.
     */
    MatchEnv(const box::Box& box, float r_max, unsigned int num_neighbors = 12);

    //! Destructor
    ~MatchEnv();

    //! Construct and return a local environment surrounding the particle indexed by i. Set the environment index to env_ind.
    Environment buildEnv(const freud::locality::NeighborList* nlist, size_t num_bonds, size_t& bond,
                         const vec3<float>* points, unsigned int i, unsigned int env_ind);

    //! Determine clusters of particles with matching environments
    /*! This is the primary interface to MatchEnv. It computes particle
     * environments and then attempts to cluster nearby particles with similar
     * environments, unless global is set true. Otherwise, it performs a
     * pairwise comparison of all particle environments to perform the match.
     * WARNING: A global search can be extremely slow.
     *
     * \param env_nlist The NeighborList used to build the environment of every particle.
     * \param nlist The NeighborList used to determine the neighbors against which 
     *              to compare environments for every particle, if hard_r = False.
     * \param threshold A unitless number that is multiply by m_r_max. This
     *                  quantity is the maximum magnitude of the vector
     *                  difference between two vectors, below which you call
     *                  them matching.  Note that ONLY values of (threshold < 2)
     *                  make any sense, since 2*r_max is the absolute
     *                  maximum difference between any two environment vectors. 
     * \param registration Controls whether we first use brute force registration to 
     *                     orient the second set of vectors such that it
     *                     minimizes the RMSD between the two sets
     * \param global If true, do an exhaustive search wherein you compare the
     *               environments of every single pair of particles in the
     *               simulation. If global is false, only compare the
     *               environments of neighboring particles.
     */
    void cluster(const freud::locality::NeighborList* env_nlist, const freud::locality::NeighborList* nlist,
                 const vec3<float>* points, unsigned int Np, float threshold,
                 bool registration = false, bool global = false);

    //! Determine whether particles match a given input motif.
    /*!
     * \param nlist A NeighborList instance.
     * \param points The points to test against the motif.
     * \param Np The number of points. 
     * \param refPoints The points characterizing the motif.
     * \param numRef The number of reference points. 
     * \param threshold A unitless number that is multiply by m_r_max. This
     *                  quantity is the maximum magnitude of the vector
     *                  difference between two vectors, below which you call
     *                  them matching.  Note that ONLY values of (threshold < 2)
     *                  make any sense, since 2*r_max is the absolute
     *                  maximum difference between any two environment vectors. 
     * \param registration Controls whether we first use brute force registration to 
     *                     orient the second set of vectors such that it
     *                     minimizes the RMSD between the two sets
     */
    void matchMotif(const freud::locality::NeighborList* nlist, const vec3<float>* points, unsigned int Np,
                    const vec3<float>* refPoints, unsigned int numRef, float threshold,
                    bool registration = false);

    //! Rotate (if registration=True) and permute the environments of all particles to minimize their RMSD wrt a given input motif.
    /*! Returns a vector of minimal RMSD values, one value per particle. NOTE
     * that this does not guarantee an absolutely minimal RMSD. It doesn't
     * figure out the optimal permutation of BOTH sets of vectors to minimize
     * the RMSD.  Rather, it just figures out the optimal permutation of the
     * second set, the vector set used in the argument below. To fully solve
     * this, we need to use the Hungarian algorithm or some other way of
     * solving the so-called assignment problem.
     *
     * \param nlist A NeighborList instance.
     * \param points The points to test against the motif.
     * \param Np The number of points. 
     * \param refPoints The points characterizing the motif.
     * \param numRef The number of reference points. 
     * \param threshold A unitless number that is multiply by m_r_max. This
     *                  quantity is the maximum magnitude of the vector
     *                  difference between two vectors, below which you call
     *                  them matching.  Note that ONLY values of (threshold < 2)
     *                  make any sense, since 2*r_max is the absolute
     *                  maximum difference between any two environment vectors. 
     * \param registration Controls whether we first use brute force registration to 
     *                     orient the second set of vectors such that it
     *                     minimizes the RMSD between the two sets
     */
    std::vector<float> minRMSDMotif(const freud::locality::NeighborList* nlist, const vec3<float>* points,
                                    unsigned int Np, const vec3<float>* refPoints, unsigned int numRef,
                                    bool registration = false);

    //! Finds a rotation matrix and the appropriate index correspondence between two environments if it exists.
    /*! If the two environments correspond, returns a std::pair of the rotation matrix that takes the
     * vectors of e2 to the vectors of e1 AND the mapping between the properly
     * indexed vectors of the environments that will make them correspond to
     * each other. If not, return a std::pair of the identity matrix AND an
     * empty map.
     *
     * \param e1 First environment.
     * \param e2 First environment.
     * \param threshold A unitless number that is multiply by m_r_max. This
     *                  quantity is the maximum magnitude of the vector
     *                  difference between two vectors, below which you call
     *                  them matching.  Note that ONLY values of (threshold < 2)
     *                  make any sense, since 2*r_max is the absolute
     *                  maximum difference between any two environment vectors. 
     * \param registration Controls whether we first use brute force registration to 
     *                     orient the second set of vectors such that it
     *                     minimizes the RMSD between the two sets
     */
    std::pair<rotmat3<float>, BiMap<unsigned int, unsigned int>>
    isSimilar(Environment& e1, Environment& e2, float threshold_sq, bool registration);

    //! Overload of the above isSimilar function that provides an easier interface to Python.
    /*! If the two environments correspond, returns a std::pair of the rotation matrix that takes the
     * vectors of e2 to the vectors of e1 AND the mapping between the properly
     * indexed vectors of the environments that will make them correspond to
     * each other. If not, return a std::pair of the identity matrix AND an
     * empty map.
     *
     * WARNING: If registration=True, then refPoints2 is CHANGED by this function.
     *
     * \param refPoints1 Points composing first environment.
     * \param refPoints2 Points composing second environment.
     * \param numRef Number of points.
     * \param threshold A unitless number that is multiply by m_r_max. This
     *                  quantity is the maximum magnitude of the vector
     *                  difference between two vectors, below which you call
     *                  them matching.  Note that ONLY values of (threshold < 2)
     *                  make any sense, since 2*r_max is the absolute
     *                  maximum difference between any two environment vectors. 
     * \param registration Controls whether we first use brute force registration to 
     *                     orient the second set of vectors such that it
     *                     minimizes the RMSD between the two sets
     */
    std::map<unsigned int, unsigned int> isSimilar(const vec3<float>* refPoints1, vec3<float>* refPoints2,
                                                   unsigned int numRef, float threshold_sq,
                                                   bool registration);

    // Get the somewhat-optimal RMSD between the (PROPERLY REGISTERED) environment e1 and the (PROPERLY REGISTERED) environment e2.
    /*! This function returns an std::pair of the rotation matrix that takes
     * the vectors of e2 to the vectors of e1 AND the mapping between the
     * properly indexed vectors of the environments that gives this RMSD.  NOTE
     * that this does not guarantee an absolutely minimal RMSD. It doesn't
     * figure out the optimal permutation of BOTH sets of vectors to minimize
     * the RMSD. Rather, it just figures out the optimal permutation of the
     * second set, the vector set used in the argument below. To fully solve
     * this, we need to use the Hungarian algorithm or some other way of
     * solving the so-called assignment problem.
     *
     * \param e1 First environment.
     * \param e2 First environment.
     * \param min_rmsd The value of the minimum RMSD (updated by reference).
     * \param registration Controls whether we first use brute force registration to 
     *                     orient the second set of vectors such that it
     *                     minimizes the RMSD between the two sets
     */
    std::pair<rotmat3<float>, BiMap<unsigned int, unsigned int>>
    minimizeRMSD(Environment& e1, Environment& e2, float& min_rmsd, bool registration);

    //! Overload of the above minimizeRMSD function that provides an easier interface to Python.
    /*! Construct the environments accordingly, and utilize minimizeRMSD() as
     * above. Arguments are pointers to interface directly with python. Return
     * a std::map (for ease of use) with the mapping between vectors refPoints1
     * and refPoints2 that gives this RMSD. 
     *
     * WARNING: If registration=True, then refPoints2 is CHANGED by this function.
     *
     * \param refPoints1 Points composing first environment.
     * \param refPoints2 Points composing second environment.
     * \param numRef Number of points.
     * \param threshold A unitless number that is multiply by m_r_max. This
     *                  quantity is the maximum magnitude of the vector
     *                  difference between two vectors, below which you call
     *                  them matching.  Note that ONLY values of (threshold < 2)
     *                  make any sense, since 2*r_max is the absolute
     *                  maximum difference between any two environment vectors. 
     * \param registration Controls whether we first use brute force registration to 
     *                     orient the second set of vectors such that it
     *                     minimizes the RMSD between the two sets
     */
    std::map<unsigned int, unsigned int> minimizeRMSD(const vec3<float>* refPoints1, vec3<float>* refPoints2,
                                                      unsigned int numRef, float& min_rmsd,
                                                      bool registration);

    //! Get a reference to the particles, indexed into clusters according to their matching local environments
    const util::ManagedArray<unsigned int> &getClusters()
    {
        return m_env_index;
    }

    //! Returns the set of vectors defining the environment indexed by i (indices culled from m_env_index)
    std::vector<vec3<float>> getEnvironment(unsigned int i)
    {
        std::map<unsigned int, std::vector<vec3<float>>>::iterator it = m_env.find(i);
        std::vector<vec3<float>> vecs = it->second;
        return vecs;
    }

    //! Returns the entire m_Np by m_k by 3 matrix of all environments for all particles
    const util::ManagedArray<vec3<float>> &getTotEnvironment()
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
        return m_num_neighbors;
    }
    unsigned int getMaxNumNeighbors()
    {
        return m_max_num_neighbors;
    }

private:

    //! Make environments from two sets of points.
    /*! This function is primarily used to offer a simpler Python interface.
    */
    std::pair<Environment, Environment> makeEnvironments(const vec3<float>* refPoints1,
                                                         vec3<float>* refPoints2, unsigned int numRef);

    //! Populate the m_env_index, m_env and m_tot_env arrays.
    /*! Renumber the clusters in the disjoint set dj from zero to num_clusters-1.
     */
    void populateEnv(EnvDisjointSet dj, bool reLabel = true);

    box::Box m_box; //!< Simulation box
    float m_r_max_sq; //!< Square of the maximum cutoff radius at which to determine local environment
    unsigned int m_num_neighbors;      //!< Default number of nearest neighbors used to determine which environments are compared
               //!< during local environment clustering.
    unsigned int m_max_num_neighbors; //!< Maximum number of neighbors in any particle's local environment. If
                         //!< hard_r=false, m_max_num_neighbors = m_num_neighbors. In the cluster method it is also possible to provide
                         //!< two separate neighborlists, one for environments and one for clustering.
    unsigned int m_Np;   //!< Last number of points computed
    unsigned int m_num_clusters; //!< Last number of local environments computed

    util::ManagedArray<unsigned int> m_env_index; //!< Cluster index determined for each particle
    std::map<unsigned int, std::vector<vec3<float>>> m_env; //!< Dictionary of (cluster id, vectors) pairs
    util::ManagedArray<vec3<float>>
        m_tot_env; //!< m_NP by m_max_num_neighbors by 3 matrix of all environments for all particles
};

}; }; // end namespace freud::environment

#endif // MATCH_ENV_H
