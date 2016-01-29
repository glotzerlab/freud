#include "MatchEnv.h"
#include "Cluster.h"
#include <map>
//#include <boost/math/special_functions.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>

namespace freud { namespace order {

// TO DO:
// 1. Create MatchEnv::isSimilar(vec1, vec2) (or maybe EnvDisjointSet::isSimilar). This should somehow both indicate to us if a set of vectors
// OF THE SAME LENGTH are similar, and if so, what the order of the vectors in vec2 should be st they match
// those in vec1 AND what the order of the vectors in vec1 should be st they match those in vec2
// 2. During MERGE, call through to vec_ind for each Environment object, and perform the same set of operations
// on each index as you do for env_ind. you have to feed it the vec_ind for the environment that is being assimilated, and then set those vec_ind.
// 3. During find, do the same thing for all vec_ind as you do for env_ind.

// Constructor for EnvDisjointSet
// Taken mostly from Cluster.cc
EnvDisjointSet::EnvDisjointSet(unsigned int num_neigh, unsigned int Np)
    : m_num_neigh(num_neigh)
    {
    rank = std::vector<unsigned int>(Np, 0);
    }

// Merge the two sets labeled by a and b.
// There is incorrect behavior if a == b or either are not set labels
// Taken mostly from Cluster.cc
void EnvDisjointSet::merge(const unsigned int a, const unsigned int b, std::map<unsigned int, unsigned int> vec_map)
    {
    assert(a < s.size() && b < s.size());
    assert(vec_map.size() == m_num_neigh)

    // if tree heights are equal, merge to a
    if (rank[a] == rank[b])
        {
        rank[a]++;
        // 1. set the environment index properly
        s[b].env_ind = a;
        // 2. set the vector indices properly
        for (unsigned int i=0; i<m_num_neigh; i++)
            {
            // STOPPED HERE.
            }


        }
    else
        {
        // merge the shorter tree to the taller one
        if (rank[a] > rank[b])
            s[b].env_ind = a;
        else
            s[a].env_ind = b;
        }
    }

// Return the set label that contains the element c
// Taken mostly from Cluster.cc
unsigned int EnvDisjointSet::find(const unsigned int c)
    {
    unsigned int r = c;

    // follow up to the root of the tree
    while (s[r].env_ind != r)
        r = s[r].env_ind;

    // path compression
    unsigned int i = c;
    while (i != r)
        {
        unsigned int j = s[i].env_ind;
        s[i].env_ind = r;
        i = j;
        }
    return r;
    }

// Constructor
MatchEnv::MatchEnv(const trajectory::Box& box, float rmax, unsigned int k)
    :m_box(box), m_rmax(rmax), m_k(k)
    {
    m_Np = 0;
    if (m_rmax < 0.0f)
        throw std::invalid_argument("rmax must be positive");
    m_rmaxsq = m_rmax * m_rmax;
    m_nn = new locality::NearestNeighbors(m_rmax, m_k);
    }

// Destructor
MatchEnv::~MatchEnv()
    {
    delete m_nn;
    }

// Build and return a local environment surrounding a particle
Environment MatchEnv::buildEnv(const vec3<float> *points, unsigned int i)
    {
    Environment ei = Environment(m_k);

    // get the neighbors
    vec3<float> p = points[i];
    boost::shared_array<unsigned int> neighbors = m_nn->getNeighbors(i);

    // loop over the neighbors
    for (unsigned int neigh_idx = 0; neigh_idx < m_k; neigh_idx++)
        {
        unsigned int j = neighbors[neigh_idx];

        // compute r between the two particles
        vec3<float> delta = m_box.wrap(p - points[j]);
        float rsq = dot(delta, delta);

        if (rsq < m_rmaxsq)
            {
            ei.addVec(delta);
            }
        }
    return ei;
    }

// Is the environment e1 similar to the environment e2?
// If so, return the mapping between the vectors of the environments that will make them correspond to each other.
// If not, return an empty map
std::map<unsigned int, unsigned int> MatchEnv::isSimilar(Environment e1, Environment e2)
    {
    std::map<unsigned int, unsigned int> vec_map;
    return vec_map;
    }

// Determine clusters of particles with matching environments
// This is taken from Cluster.cc and SolLiq.cc and LocalQlNear.cc
void MatchEnv::compute(const vec3<float> *points, unsigned int Np)
    {
    assert(points);
    assert(Np > 0);

    // reallocate the m_env_index array if the size doesn't match the last one
    if (Np != m_Np)
        m_env_index = boost::shared_array<unsigned int>(new unsigned int[Np]);

    m_Np = Np;

    // initialize the neighbor list
    m_nn->compute(m_box, points, m_Np, points, m_Np);

    // create a disjoint set where all particles belong in their own cluster
    EnvDisjointSet dj(m_k, m_Np);

    // add all the environments to the set
    for (unsigned int i = 0; i < m_Np; i++)
        {
        Environment ei = buildEnv(points, i);
        dj.s.push_back(ei);
        }

    // loop through points
    for (unsigned int i = 0; i < m_Np; i++)
        {

        // 1. Get all the neighbors
        vec3<float> p = points[i];
        boost::shared_array<unsigned int> neighbors = m_nn->getNeighbors(i);

        // loop over the neighbors
        for (unsigned int neigh_idx = 0; neigh_idx < m_k; neigh_idx++)
            {
            unsigned int j = neighbors[neigh_idx];

            if (i != j)
                {
                std::map<unsigned int, unsigned int> vec_map = isSimilar(dj.s[i], dj.s[j]);
                // if the mapping between the vectors of the environments is NOT empty, then the environments
                // are similar. so merge them.
                if (!vec_map.empty())
                    {
                    // merge the two sets using the disjoint set
                    unsigned int a = dj.find(i);
                    unsigned int b = dj.find(j);
                    if (a != b)
                        dj.merge(a,b,vec_map);
                    }
                }
            }
        }

    // done looping over points. All clusters are now determined. Renumber them from zero to num_clusters-1.
    std::map<unsigned int, unsigned int> label_map;

    // loop over all particles
    unsigned int cur_set = 0;
    for (unsigned int i = 0; i < m_Np; i++)
        {
        unsigned int c = dj.find(i);

        // insert the set into the mapping if we haven't seen it before
        if (label_map.count(c) == 0)
            {
            label_map[c] = cur_set;
            cur_set++;
            }

        // label this particle in m_env_index
        m_env_index[i] = label_map[c];
        }
    }

}; }; // end namespace freud::match_env;
