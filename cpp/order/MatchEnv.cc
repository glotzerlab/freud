#include <cstdio>
#include "MatchEnv.h"
#include "Cluster.h"

namespace freud { namespace order {

// Constructor for EnvDisjointSet
// Taken partially from Cluster.cc
EnvDisjointSet::EnvDisjointSet(unsigned int num_neigh, unsigned int Np)
    : m_num_neigh(num_neigh)
    {
    rank = std::vector<unsigned int>(Np, 0);
    }

// Merge the two sets that elements a and b belong to.
// Taken partially from Cluster.cc
// The vec_map must be a bimap of vector indices where those of set a are on the left and those of set b are on the right.
void EnvDisjointSet::merge(const unsigned int a, const unsigned int b, boost::bimap<unsigned int, unsigned int> vec_map)
    {
    assert(a < s.size() && b < s.size());
    assert(vec_map.size() == m_num_neigh);

    // if tree heights are equal, merge b to a
    if (rank[s[a].env_ind] == rank[s[b].env_ind])
        {
        // 0. Get the ENTIRE set that corresponds to head_b.
        // First make a copy of the b environment so we don't get all mixed up.
        std::vector<unsigned int> old_b_vec_ind = s[b].vec_ind;
        unsigned int head_b = find(b);
        std::vector<unsigned int> m_set = findSet(head_b);
        for (unsigned int n = 0; n < m_set.size(); n++)
            {
            // Go through the entire tree/set.
            unsigned int node = m_set[n];
            // Make a copy of the old set of vector indices for this particular node. This is complicated and weird.
            std::vector<unsigned int> old_node_vec_ind = s[node].vec_ind;

            // Set the vector indices properly.
            // Iterate over the vector indices of a.
            // Take the LEFT MAP view of the a<->b bimap.
            // Find the value of b_ind that corresponds to the value of a_ind, and set it properly.
            for (unsigned int i=0; i<m_num_neigh; i++)
                {
                unsigned int a_ind = s[a].vec_ind[i];
                boost::bimap<unsigned int, unsigned int>::left_const_iterator it = vec_map.left.find(a_ind);
                unsigned int b_ind = it->second;

                // Here's the proper setting: find the location of b_ind in the current vec_ind vector, then bind the corresponding vec_ind to a_ind. (in the same location as a_ind.)
                // For node=b, this is the same as s[b].vec_ind[i] = b_ind.
                std::vector<unsigned int>::iterator b_it = std::find(old_b_vec_ind.begin(), old_b_vec_ind.end(), b_ind);
                unsigned int b_ind_position = b_it - old_b_vec_ind.begin();
                s[node].vec_ind[i] = old_node_vec_ind[b_ind_position];
                }

            // set the environment index properly
            s[node].env_ind = s[a].env_ind;

            // we've added another leaf to the tree or whatever the lingo is.
            rank[s[a].env_ind]++;
            }
        }
    else
        {
        // merge the shorter tree to the taller one
        if (rank[s[a].env_ind] > rank[s[b].env_ind])
            {
            // 0. Get the ENTIRE set that corresponds to head_b.
            // First make a copy of the b environment so we don't get all mixed up.
            std::vector<unsigned int> old_b_vec_ind = s[b].vec_ind;
            unsigned int head_b = find(b);
            std::vector<unsigned int> m_set = findSet(head_b);
            for (unsigned int n = 0; n < m_set.size(); n++)
                {
                // Go through the entire tree/set.
                unsigned int node = m_set[n];
                // Make a copy of the old set of vector indices for this particular node. This is complicated and weird.
                std::vector<unsigned int> old_node_vec_ind = s[node].vec_ind;

                // Set the vector indices properly.
                // Iterate over the vector indices of a.
                // Take the LEFT MAP view of the a<->b bimap.
                // Find the value of b_ind that corresponds to the value of a_ind, and set it properly.
                for (unsigned int i=0; i<m_num_neigh; i++)
                    {
                    unsigned int a_ind = s[a].vec_ind[i];
                    boost::bimap<unsigned int, unsigned int>::left_const_iterator it = vec_map.left.find(a_ind);
                    unsigned int b_ind = it->second;

                    // Here's the proper setting: find the location of b_ind in the current vec_ind vector, then bind the corresponding vec_ind to a_ind. (in the same location as a_ind.)
                    // For node=b, this is the same as s[b].vec_ind[i] = b_ind.
                    std::vector<unsigned int>::iterator b_it = std::find(old_b_vec_ind.begin(), old_b_vec_ind.end(), b_ind);
                    unsigned int b_ind_position = b_it - old_b_vec_ind.begin();
                    s[node].vec_ind[i] = old_node_vec_ind[b_ind_position];
                    }

                // set the environment index properly
                s[node].env_ind = s[a].env_ind;

                // we've added another leaf to the tree or whatever the lingo is.
                rank[s[a].env_ind]++;
                }
            }
        else
            {
            // 0. Get the ENTIRE set that corresponds to head_a.
            // First make a copy of the a environment so we don't get all mixed up.
            std::vector<unsigned int> old_a_vec_ind = s[a].vec_ind;
            unsigned int head_a = find(a);
            std::vector<unsigned int> m_set = findSet(head_a);
            for (unsigned int n = 0; n < m_set.size(); n++)
                {
                // Go through the entire tree/set.
                unsigned int node = m_set[n];
                // Make a copy of the old set of vector indices for this particular node. This is complicated and weird.
                std::vector<unsigned int> old_node_vec_ind = s[node].vec_ind;

                // Set the vector indices properly.
                // Iterate over the vector indices of b.
                // Take the RIGHT MAP view of the a<->b bimap.
                // Find the value of a_ind that corresponds to the value of b_ind, and set it properly.
                for (unsigned int i=0; i<m_num_neigh; i++)
                    {
                    unsigned int b_ind = s[b].vec_ind[i];
                    boost::bimap<unsigned int, unsigned int>::right_const_iterator it = vec_map.right.find(b_ind);
                    unsigned int a_ind = it->second;

                    // Here's the proper setting: find the location of a_ind in the current vec_ind vector, then bind the corresponding vec_ind to b_ind. (in the same location as b_ind.)
                    // For node=a, this is the same as s[a].vec_ind[i] = a_ind.
                    std::vector<unsigned int>::iterator a_it = std::find(old_a_vec_ind.begin(), old_a_vec_ind.end(), a_ind);
                    unsigned int a_ind_position = a_it - old_a_vec_ind.begin();
                    s[node].vec_ind[i] = old_node_vec_ind[a_ind_position];
                    }

                // set the environment index properly
                s[node].env_ind = s[b].env_ind;

                // we've added another leaf to the tree or whatever the lingo is.
                rank[s[b].env_ind]++;
                }
            }
        }
    }

// Return the set label that contains the element c
// Taken mostly from Cluster.cc
unsigned int EnvDisjointSet::find(const unsigned int c)
    {
    unsigned int r = c;

    // follow up to the head of the tree
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

// Return ALL nodes in the tree that correspond to the head index m.
// Values returned: the actual locations of the nodes in s. (i.e. if i is returned, the node is accessed by s[i]).
// If environment m doesn't exist as a HEAD in the set, throw an error.
std::vector<unsigned int> EnvDisjointSet::findSet(const unsigned int m)
    {
    assert(s.size() > 0);
    bool invalid_ind = true;

    std::vector<unsigned int> m_set;

    // this is wildly inefficient
    for (unsigned int i = 0; i < s.size(); i++)
        {
        // get the head environment index
        unsigned int head_env = find(s[i].env_ind);
        // if we are part of the environment m, add the vectors to m_set
        if (head_env == m)
            {
            m_set.push_back(i);
            invalid_ind = false;
            }
        }

    if (invalid_ind)
        {
        fprintf(stderr, "m is %d\n", m);
        throw std::invalid_argument("m must be a head index in the environment set!");
        }

    return m_set;
    }

// Get the vectors corresponding to environment head index m. Vectors are averaged over all members of the environment cluster.
// If environment m doesn't exist as a HEAD in the set, throw an error.
boost::shared_array<vec3<float> > EnvDisjointSet::getAvgEnv(const unsigned int m)
    {
    assert(s.size() > 0);
    bool invalid_ind = true;
    bool single_particle = true;

    boost::shared_array<vec3<float> > env(new vec3<float> [m_num_neigh] );
    for (unsigned int n = 0; n < m_num_neigh; n++)
        {
        env[n] = vec3<float>(0.0,0.0,0.0);
        }
    float N = float(0);

    // loop over all the environments in the set
    for (unsigned int i = 0; i < s.size(); i++)
        {
        // if this environment is NOT a ghost (i.e. non-physical):
        if (s[i].ghost == false)
            {
            // get the head environment index
            unsigned int head_env = find(s[i].env_ind);
            // if we are part of the environment m, add the vectors to env
            if (head_env == m)
                {
                if (!single_particle)
                    {
                    assert(s[i].vec_ind.size() == m_num_neigh);
                    assert(s[i].vecs.size() == m_num_neigh);
                    }
                // loop through the vectors, getting them properly indexed
                // add them to env
                for (unsigned int j = 0; j < s[i].vecs.size(); j++)
                    {
                    unsigned int proper_ind = s[i].vec_ind[j];
                    env[j] += s[i].vecs[proper_ind];
                    }
                N += float(1);
                single_particle=false;
                invalid_ind = false;
                }
            }
        }

    if (invalid_ind)
        {
        fprintf(stderr, "m is %d\n", m);
        throw std::invalid_argument("m must be a head index in the environment set!");
        }

    else
        {
        // loop through the vectors in env now, dividing by the total number of contributing particle environments to make an average
        for (unsigned int n = 0; n < m_num_neigh; n++)
            {
            vec3<float> normed = env[n]/N;
            env[n] = normed;
            }
        }
    return env;
    }

// Get the vectors corresponding to index m in the dj set
// If index m doesn't exist in the set, throw an error.
std::vector<vec3<float> > EnvDisjointSet::getIndividualEnv(const unsigned int m)
    {
    assert(s.size() > 0);
    if (m >= s.size())
        {
        fprintf(stderr, "m is %d\n", m);
        throw std::invalid_argument("m is indexing into the environment set. It must be less than the size of the set!");
        }

    std::vector<vec3<float> > env;
    for (unsigned int n = 0; n < m_num_neigh; n++)
        {
        env.push_back(vec3<float>(0.0,0.0,0.0));
        }

    // loop through the vectors, getting them properly indexed
    // add them to env
    for (unsigned int j = 0; j < s[m].vecs.size(); j++)
        {
        unsigned int proper_ind = s[m].vec_ind[j];
        env[j] += s[m].vecs[proper_ind];
        }

    return env;
    }

// Constructor
MatchEnv::MatchEnv(const trajectory::Box& box, float rmax, unsigned int k)
    :m_box(box), m_rmax(rmax), m_k(k)
    {
    m_Np = 0;
    m_num_clusters = 0;
    if (m_rmax < 0.0f)
        throw std::invalid_argument("rmax must be positive!");
    m_rmaxsq = m_rmax * m_rmax;
    m_nn = new locality::NearestNeighbors(m_rmax, m_k);
    }

// Destructor
MatchEnv::~MatchEnv()
    {
    delete m_nn;
    }

// Build and return a local environment surrounding a particle.
// Label its environment with env_ind.
Environment MatchEnv::buildEnv(const vec3<float> *points, unsigned int i, unsigned int env_ind)
    {
    Environment ei = Environment(m_k);
    // set the environment index equal to the particle index
    ei.env_ind = env_ind;

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
// The threshold is a unitless number, which we multiply by the length scale of the MatchEnv instance, rmax.
// This quantity is the maximum squared magnitude of the vector difference between two vectors, below which you call them matching.
boost::bimap<unsigned int, unsigned int> MatchEnv::isSimilar(Environment e1, Environment e2, float threshold_sq)
    {
    std::vector< vec3<float> > v1 = e1.vecs;
    std::vector< vec3<float> > v2 = e2.vecs;
    boost::bimap<unsigned int, unsigned int> vec_map;

    // Inelegant: if either vector set does not have m_k vectors in it (i.e. maybe we are at a surface),
    // just return an empty map for now since the 1-1 bimapping will be too weird in this case.
    if (v1.size() != m_k) { return vec_map; }
    if (v2.size() != m_k) { return vec_map; }

    // compare all combinations of vectors
    for (unsigned int i = 0; i < m_k; i++)
        {
        for (unsigned int j = 0; j < m_k; j++)
            {
            vec3<float> delta = v1[i] - v2[j];
            // delta = m_box.wrap(delta);
            float rsq = dot(delta, delta);
            if (rsq < threshold_sq*m_rmaxsq)
                {
                // these vectors are deemed "matching"
                // since this is a bimap, this (i,j) pair is only inserted if j has not already been assigned an i pairing.
                // (ditto with i not being assigned a j pairing)
                vec_map.insert(boost::bimap<unsigned int, unsigned int>::value_type(i, j));
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

// Get the optimal RMSD between the set of vectors v1 and the set of vectors v2
// Populate the empty boost::bimap with the mapping between vectors v1 and v2 that gives this optimal RMSD.
// If vectors have a different number of elements, return -1.0 and force the bimap to be empty
// A little of the logic here is taken from Paul's procrustes library, in his original brute_force.h AlignedRMSD() method. Thanks Paul!
double MatchEnv::getMinRMSD(const std::vector<vec3<float> >& v1, const std::vector<vec3<float> >& v2, boost::bimap<unsigned int, unsigned int>& m)
    {
    boost::bimap<unsigned int, unsigned int> vec_map;

    // If the vectors are two different sizes, force the map to be empty since it can never be 1-1.
    // Return Min RMSD = -1.
    if (v1.size() != v2.size())
        {
        m = vec_map;
        return -1.0;
        }

    // compare all combinations of vectors
    for (unsigned int i = 0; i < v1.size(); i++)
        {
        double min_rsq = -1.0;
        for (unsigned int j = 0; j < v2.size(); j++)
            {
            vec3<float> delta = v1[i] - v2[j];
            float rsq = dot(delta, delta);
            if (rsq < min_rsq || min_rsq < 0.0)
                {
                min_rsq = rsq;
                }
            }
        }

    // THIS DOESNT QUITE WORK: it depends on the order with which you loop through v1, doesn't it? for that matter so does isSimilar. imagine 2 sets of 2 vectors each, A and B.
    // vector B1 is 0.5 distance away from A1 and A2, and vector B2 is 0.5 distance away from A2 and 1.0 distance away from A1. Let's say we look at A2 first. we find the closest vector is B1.
    // Then we look at A1 and find the closest vector has to be B2. The RMSD here is 0.5 [0.5 + 1.0]. This isn't optimal though. The optimal RMSD is obviously 0.5 [0.5 + 0.5]. Were we to measure
    // isSimilar with a threshold of 0.6 difference between each vector, we would say A and B were NOT similar in the first case, even though they should be called as similar.
    // Does Paul's AlignedRMSDTree take care of this?

    }

// Overload: is the set of vectors refPoints1 similar to the set of vectors refPoints2?
// Construct the environments accordingly, and utilize isSimilar() as above.
// Return a std map for ease of use.
std::map<unsigned int, unsigned int> MatchEnv::isSimilar(const vec3<float> *refPoints1, const vec3<float> *refPoints2, unsigned int numRef, float threshold_sq)
    {
    assert(refPoints1);
    assert(refPoints2);
    assert(numRef == m_k);

    // create the environment characterized by refPoints1. Index it as 0.
    // set the IGNORE flag to true, since this is not an environment we have actually encountered in the simulation.
    Environment e0 = Environment(m_k);
    e0.env_ind = 0;
    e0.ghost = true;

    // create the environment characterized by refPoints2. Index it as 1.
    // set the IGNORE flag to true again.
    Environment e1 = Environment(m_k);
    e1.env_ind = 1;
    e1.ghost = true;

    // loop through all the vectors in refPoints1 and refPoints2 and add them to the environments.
    // wrap all the vectors back into the box. I think this is necessary since all the vectors
    // that will be added to actual particle environments will be wrapped into the box as well.
    for (unsigned int i = 0; i < numRef; i++)
        {
        vec3<float> p0 = m_box.wrap(refPoints1[i]);
        vec3<float> p1 = m_box.wrap(refPoints2[i]);
        e0.addVec(p0);
        e1.addVec(p1);
        }

    // call isSimilar for e0 and e1
    boost::bimap<unsigned int, unsigned int> vec_map = isSimilar(e0, e1, threshold_sq);

    // convert to a std::map
    // the lamest.
    // from stackoverflow.com/questions/20667187/convert-boostbimap-to-stdmap
    std::map<unsigned int, unsigned int> std_vec_map;
    for (boost::bimap<unsigned int, unsigned int>::const_iterator it = vec_map.begin(); it != vec_map.end(); ++it)
        {
        std_vec_map[it->left] = it->right;
        }

    // return the vector map
    return std_vec_map;
    }

// Determine clusters of particles with matching environments
// This is taken from Cluster.cc and SolLiq.cc and LocalQlNear.cc
void MatchEnv::cluster(const vec3<float> *points, unsigned int Np, float threshold)
    {
    assert(points);
    assert(Np > 0);
    assert(threshold > 0);

    // reallocate the m_env_index array for safety
    m_env_index = boost::shared_array<unsigned int>(new unsigned int[Np]);
    // also reallocate the m_tot_env array
    unsigned int array_size = Np*m_k;
    m_tot_env = boost::shared_array<vec3<float> >(new vec3<float>[array_size]);

    m_Np = Np;
    float m_threshold_sq = threshold*threshold;

    // initialize the neighbor list
    m_nn->compute(m_box, points, m_Np, points, m_Np);

    // create a disjoint set where all particles belong in their own cluster
    EnvDisjointSet dj(m_k, m_Np);

    // add all the environments to the set
    // take care, here: set things up s.t. the env_ind of every environment matches its location in the disjoint set.
    // if you don't do this, things will get screwy.
    for (unsigned int i = 0; i < m_Np; i++)
        {
        Environment ei = buildEnv(points, i, i);
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
                boost::bimap<unsigned int, unsigned int> vec_map = isSimilar(dj.s[i], dj.s[j], m_threshold_sq);
                // if the mapping between the vectors of the environments is NOT empty, then the environments
                // are similar. so merge them.
                if (!vec_map.empty())
                    {
                    // merge the two sets using the disjoint set
                    unsigned int a = dj.find(i);
                    unsigned int b = dj.find(j);
                    if (a != b)
                        dj.merge(i,j,vec_map);
                    }
                }
            }
        }

    // done looping over points. All clusters are now determined. Renumber them from zero to num_clusters-1.
    populateEnv(dj, true);
    }

//! Determine whether particles match a given input motif, characterized by refPoints (of which there are numRef)
void MatchEnv::matchMotif(const vec3<float> *points, unsigned int Np, const vec3<float> *refPoints, unsigned int numRef, float threshold)
    {
    assert(points);
    assert(refPoints);
    assert(numRef == m_k);
    assert(Np > 0);
    assert(threshold > 0);

    // reallocate the m_env_index array for safety
    m_env_index = boost::shared_array<unsigned int>(new unsigned int[Np]);
    // also reallocate the m_tot_env array
    unsigned int array_size = Np*m_k;
    m_tot_env = boost::shared_array<vec3<float> >(new vec3<float>[array_size]);

    m_Np = Np;
    float m_threshold_sq = threshold*threshold;

    // initialize the neighbor list
    m_nn->compute(m_box, points, m_Np, points, m_Np);

    // create a disjoint set where all particles belong in their own cluster.
    // this has to have ONE MORE environment than there are actual particles, because we're inserting the motif into it.
    EnvDisjointSet dj(m_k, m_Np+1);

    // create the environment characterized by refPoints. Index it as 0.
    // set the IGNORE flag to true, since this is not an environment we have actually encountered in the simulation.
    Environment e0 = Environment(m_k);
    e0.env_ind = 0;
    e0.ghost = true;

    // loop through all the vectors in refPoints and add them to the environment.
    // wrap all the vectors back into the box. I think this is necessary since all the vectors
    // that will be added to actual particle environments will be wrapped into the box as well.
    for (unsigned int i = 0; i < numRef; i++)
        {
        vec3<float> p = m_box.wrap(refPoints[i]);
        e0.addVec(p);
        }

    // add this environment to the set
    dj.s.push_back(e0);

    // loop through the particles and add their environments to the set
    // take care, here: set things up s.t. the env_ind of every environment matches its location in the disjoint set.
    // if you don't do this, things will get screwy.
    for (unsigned int i = 0; i < m_Np; i++)
        {
        unsigned int dummy = i+1;
        Environment ei = buildEnv(points, i, dummy);
        dj.s.push_back(ei);

        // if the environment matches e0, merge it into the e0 environment set
        boost::bimap<unsigned int, unsigned int> vec_map = isSimilar(dj.s[0], dj.s[dummy], m_threshold_sq);
        // if the mapping between the vectors of the environments is NOT empty, then the environments are similar.
        if (!vec_map.empty())
            {
            dj.merge(0, dummy, vec_map);
            }
        }

    // DON'T renumber the clusters in the disjoint set from zero to num_clusters-1.
    // The way I have set it up here, the "0th" cluster is the one that matches the motif.
    populateEnv(dj, false);

    }

//! Populate the m_env_index, m_env and m_tot_env arrays.
//! Renumber the clusters in the disjoint set dj from zero to num_clusters-1, if that is called.
void MatchEnv::populateEnv(EnvDisjointSet dj, bool reLabel)
    {
    std::map<unsigned int, unsigned int> label_map;

    // loop over all environments
    unsigned int label_ind;
    unsigned int cur_set = 0;
    unsigned int particle_ind = 0;
    for (unsigned int i = 0; i < dj.s.size(); i++)
        {
        // only count this if the environment is physical
        if (dj.s[i].ghost == false)
            {
            // grab the set of vectors that define this individual environment
            std::vector<vec3<float> > part_vecs = dj.getIndividualEnv(i);

            unsigned int c = dj.find(i);
            // insert the set into the mapping if we haven't seen it before.
            // also grab the vectors that define the set and insert them into m_env
            if (label_map.count(c) == 0)
                {
                label_map[c] = cur_set;
                boost::shared_array<vec3<float> > vecs = dj.getAvgEnv(c);

                if (reLabel == true) { label_ind = label_map[c]; }
                else { label_ind = c; }

                m_env[label_ind] = vecs;

                cur_set++;
                }

            if (reLabel == true) { label_ind = label_map[c]; }
            else { label_ind = c; }

            // label this particle in m_env_index
            m_env_index[particle_ind] = label_ind;
            // add the particle environment to m_tot_env
            // get a pointer to the start of m_tot_env
            vec3<float> *start = m_tot_env.get();
            // loop through part_vecs and add them
            for (unsigned int m = 0; m < part_vecs.size(); m++)
                {
                unsigned int index = particle_ind*m_k + m;
                start[index] = part_vecs[m];
                }
            particle_ind++;
            }
        }

    // specify the number of cluster environments
    m_num_clusters = cur_set;
    }

}; }; // end namespace freud::match_env;
