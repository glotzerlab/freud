#include "MatchEnv.h"
#include "Cluster.h"

namespace freud { namespace order {

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
// The vec_map must be a bimap of vector indices where those of set a are on the left and those of set b are
// on the right.
void EnvDisjointSet::merge(const unsigned int a, const unsigned int b, boost::bimap<unsigned int, unsigned int> vec_map)
    {
    assert(a < s.size() && b < s.size());
    assert(vec_map.size() == m_num_neigh);

    // if tree heights are equal, merge to a
    if (rank[a] == rank[b])
        {
        rank[a]++;
        // 1. set the environment index properly
        s[b].env_ind = s[a].env_ind;
        // 2. set the vector indices properly.
        // Iterate over the vector indices of a.
        // Take the LEFT MAP view of the a<->b bimap.
        // Find the value of b_ind that corresponds to the value of a_ind, and set it for b.
        for (unsigned int i=0; i<m_num_neigh; i++)
            {
            unsigned int a_ind = s[a].vec_ind[i];
            boost::bimap<unsigned int, unsigned int>::left_const_iterator it = vec_map.left.find(a_ind);
            unsigned int b_ind = it->second;
            s[b].vec_ind[i] = b_ind;
            }
        }
    else
        {
        // merge the shorter tree to the taller one
        if (rank[a] > rank[b])
            {
            s[b].env_ind = s[a].env_ind;
            // Merge to a.
            for (unsigned int i=0; i<m_num_neigh; i++)
                {
                unsigned int a_ind = s[a].vec_ind[i];
                boost::bimap<unsigned int, unsigned int>::left_const_iterator it = vec_map.left.find(a_ind);
                unsigned int b_ind = it->second;
                s[b].vec_ind[i] = b_ind;
                }
            }
        else
            {
            s[a].env_ind = s[b].env_ind;
            // Merge to b.
            for (unsigned int i=0; i<m_num_neigh; i++)
                {
                unsigned int b_ind = s[b].vec_ind[i];
                boost::bimap<unsigned int, unsigned int>::right_const_iterator it = vec_map.right.find(b_ind);
                unsigned int a_ind = it->second;
                s[a].vec_ind[i] = a_ind;
                }
            }
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
    // set the environment index equal to the particle index
    ei.env_ind = i;

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
// If not, return an empty map.
// The threshold is the maximum squared magnitude of the vector difference between two vectors, below which you call them matching.
boost::bimap<unsigned int, unsigned int> MatchEnv::isSimilar(Environment e1, Environment e2, float threshold_sq)
    {
    std::vector< vec3<float> > v1 = e1.vecs;
    std::vector< vec3<float> > v2 = e2.vecs;
    boost::bimap<unsigned int, unsigned int> vec_map;

    // Inelegant: if either vector set does not have m_k vectors in it (i.e. maybe we are at a surface),
    // just return an empty map for now since the 1-1 bimapping will be too weird in this case.
    if (v1.size() != m_k) { return vec_map; }
    if (v2.size() != m_k) { return vec_map; }

    // compare all iterations of vectors
    for (unsigned int i = 0; i < m_k; i++)
        {
        for (unsigned int j = 0; j < m_k; j++)
            {
            vec3<float> delta = v1[i] - v2[j];

            delta = m_box.wrap(delta);
            float rsq = dot(delta, delta);
            // std::cout<<rsq<<std::endl;

            if (rsq < threshold_sq)
                {
                // these vectors are deemed "matching"
                // since this is a bimap, this (i,j) pair is only inserted if j has not already been assigned an i pairing.
                // (ditto with i not being assigned a j pairing)
                vec_map.insert(boost::bimap<unsigned int, unsigned int>::value_type(i,j));
                }
            }
        }

    // if every vector has been paired with every other vector, return this bimap
    if (vec_map.size() == m_k)
        {
        return vec_map;
        }
    // otherwise, return an empty bimap
    else
        {
        boost::bimap<unsigned int, unsigned int> empty_map;
        return empty_map;
        }
    }

// Determine clusters of particles with matching environments
// This is taken from Cluster.cc and SolLiq.cc and LocalQlNear.cc
void MatchEnv::compute(const vec3<float> *points, unsigned int Np, float threshold)
    {
    assert(points);
    assert(Np > 0);
    assert(threshold > 0);

    // reallocate the m_env_index array if the size doesn't match the last one
    if (Np != m_Np)
        m_env_index = boost::shared_array<unsigned int>(new unsigned int[Np]);

    m_Np = Np;
    float m_threshold_sq = threshold*threshold;

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
            // std::cout<<j<<std::endl;

            if (i != j)
                {
                boost::bimap<unsigned int, unsigned int> vec_map = isSimilar(dj.s[i], dj.s[j], m_threshold_sq);
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

        // insert the set into the mapping if we haven't seen it before.
        // also grab the vectors that define the set and insert them into m_env
        if (label_map.count(c) == 0)
            {
            label_map[c] = cur_set;
            
            m_env[c] = vecs;

            cur_set++;
            }

        // label this particle in m_env_index
        m_env_index[i] = label_map[c];
        }
    }

}; }; // end namespace freud::match_env;
