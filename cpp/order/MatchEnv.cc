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

    // if (s[a].vecs[7].x > 0) {std::cout<<"merging "<<a<<" and "<<b<<std::endl;}
    // if (s[a].vecs[7].x > 0) {std::cout<<"env ind "<<s[a].env_ind<<" and "<<s[b].env_ind<<std::endl;}
    // if (a==4 && b==5) {std::cout<<"merging "<<a<<" and "<<b<<std::endl;}
    // if (a==4 && b==5) {std::cout<<"env ind "<<s[a].env_ind<<" and "<<s[b].env_ind<<std::endl;}

    // if tree heights are equal, merge b to a
    if (rank[s[a].env_ind] == rank[s[b].env_ind])
        {
        // 0. Head up the tree, starting at the current environment.
        // First make a copy of the current environment so we don't get all mixed up.
        std::vector<unsigned int> old_b_vec_ind = s[b].vec_ind;
        unsigned int r = b;
        unsigned int head_b = find(b);
        bool my_own_head = false;
        while (!my_own_head)
            {
            // if (first_time==false) {std::cout<<"we're going up the tree!"<<std::endl;}
            // if (first_time==false) {std::cout<<"particle_ind: "<<r<<" env_ind: "<<s[r].env_ind<<std::endl;}
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
                // For r=b, the first time, this is the same as s[b].vec_ind[i] = b_ind.
                std::vector<unsigned int>::iterator b_it = std::find(old_b_vec_ind.begin(), old_b_vec_ind.end(), b_ind);
                unsigned int b_ind_position = b_it - old_b_vec_ind.begin();
                // std::cout<<"b_ind: "<<b_ind<<std::endl;
                // std::cout<<"b_ind_position: "<<b_ind_position<<std::endl;
                s[r].vec_ind[i] = s[r].vec_ind[b_ind_position];
                }

            // prepare to go up the tree
            unsigned int j = s[r].env_ind;
            // set the environment index properly
            s[r].env_ind = s[a].env_ind;

            if (r == j)
                {
                my_own_head = true;
                }
            else
                {
                r = j;
                }

            // we've added another leaf to the tree or whatever the lingo is.
            rank[s[a].env_ind]++;
            }
        }
    else
        {
        // merge the shorter tree to the taller one
        if (rank[s[a].env_ind] > rank[s[b].env_ind])
            {
            // 0. Head up the tree, starting at the current environment.
            // First make a copy of the current environment so we don't get all mixed up.
            std::vector<unsigned int> old_b_vec_ind = s[b].vec_ind;
            unsigned int r = b;
            unsigned int head_b = find(b);
            bool my_own_head = false;
            while (!my_own_head)
                {
                // if (first_time==false) {std::cout<<"we're going up the tree!"<<std::endl;}
                // if (first_time==false) {std::cout<<"particle_ind: "<<r<<" env_ind: "<<s[r].env_ind<<std::endl;}
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
                    // For r=b, the first time, this is the same as s[b].vec_ind[i] = b_ind.
                    std::vector<unsigned int>::iterator b_it = std::find(old_b_vec_ind.begin(), old_b_vec_ind.end(), b_ind);
                    unsigned int b_ind_position = b_it - old_b_vec_ind.begin();
                    // std::cout<<"b_ind: "<<b_ind<<std::endl;
                    // std::cout<<"b_ind_position: "<<b_ind_position<<std::endl;
                    s[r].vec_ind[i] = s[r].vec_ind[b_ind_position];
                    }

                // prepare to go up the tree
                unsigned int j = s[r].env_ind;
                // set the environment index properly
                s[r].env_ind = s[a].env_ind;

                if (r == j)
                    {
                    my_own_head = true;
                    }
                else
                    {
                    r = j;
                    }

                // we've added another leaf to the tree or whatever the lingo is.
                rank[s[a].env_ind]++;
                }
            }
        else
            {
            // 0. Head up the tree, starting at the current environment.
            // First make a copy of the current environment so we don't get all mixed up.
            std::vector<unsigned int> old_a_vec_ind = s[a].vec_ind;
            unsigned int r = a;
            unsigned int head_a = find(a);
            bool my_own_head = false;
            while (!my_own_head)
                {
                // if (first_time==false) {std::cout<<"we're going up the tree!"<<std::endl;}
                // if (first_time==false) {std::cout<<"particle_ind: "<<r<<" env_ind: "<<s[r].env_ind<<std::endl;}
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
                    // For r=a, the first time, this is the same as s[a].vec_ind[i] = a_ind.
                    std::vector<unsigned int>::iterator a_it = std::find(old_a_vec_ind.begin(), old_a_vec_ind.end(), a_ind);
                    unsigned int a_ind_position = a_it - old_a_vec_ind.begin();
                    // std::cout<<"a_ind: "<<a_ind<<std::endl;
                    // std::cout<<"a_ind_position: "<<a_ind_position<<std::endl;
                    s[r].vec_ind[i] = s[r].vec_ind[a_ind_position];
                    }

                // prepare to go up the tree
                unsigned int j = s[r].env_ind;
                // set the environment index properly
                s[r].env_ind = s[b].env_ind;

                if (r == j)
                    {
                    my_own_head = true;
                    }
                else
                    {
                    r = j;
                    }

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

    // follow up to the root of the tree
    while (s[r].env_ind != r)
        r = s[r].env_ind;

    // // path compression
    // unsigned int i = c;
    // while (i != r)
    //     {
    //     unsigned int j = s[i].env_ind;
    //     s[i].env_ind = r;
    //     i = j;
    //     }
    return r;
    }

//! Get the vectors corresponding to environment root index m
//! If environment i doesn't exist as a ROOT in the set, throw an error.
boost::shared_array<vec3<float> > EnvDisjointSet::getEnv(const unsigned int m)
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
        // if we have been told NOT to ignore this environment:
        if (s[i].ignore == false)
            {
            // get the root environment index
            unsigned int root_env = find(s[i].env_ind);
            // if we are part of the environment m, add the vectors to env
            if (root_env == m)
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
                    // if (proper_ind==7) { std::cout<<"env "<<m<<" vec 7: "<<s[i].vecs[proper_ind].x<<" "<<s[i].vecs[proper_ind].y<<" "<<s[i].vecs[proper_ind].z<<std::endl; }
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
        throw std::invalid_argument("m must be a root index in the environment set!");
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
// Label its environment with env_ind
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
        // be sure to tie together the PROPER vector indices, in case any index re-ordering has taken place
        // unsigned int proper_i = e1.vec_ind[i];
        // if (e1.env_ind==4 && proper_i==7) { std::cout<<"env4 vec 7: "<<v1[proper_i].x<<" "<<v1[proper_i].y<<" "<<v1[proper_i].z<<std::endl; }
        for (unsigned int j = 0; j < m_k; j++)
            {
            // unsigned int proper_j = e2.vec_ind[j];
            vec3<float> delta = v1[i] - v2[j];
            // vec3<float> delta = v1[proper_i] - v2[proper_j];
            // if (e1.env_ind==0 && e2.env_ind==1) { std::cout<<"v1: "<<v1[proper_i].x<<" "<<v1[proper_i].y<<" "<<v1[proper_i].z<<std::endl; }
            // if (e1.env_ind==4 && proper_i==7) { std::cout<<"v2: "<<v2[proper_j].x<<" "<<v2[proper_j].y<<" "<<v2[proper_j].z<<std::endl; }
            // if (e1.env_ind==4 && proper_i==7) { std::cout<<"delta: "<<delta.x<<" "<<delta.y<<" "<<delta.z<<std::endl; }
            // delta = m_box.wrap(delta);
            float rsq = dot(delta, delta);
            if (rsq < threshold_sq)
                {
                // these vectors are deemed "matching"
                // since this is a bimap, this (i,j) pair is only inserted if j has not already been assigned an i pairing.
                // (ditto with i not being assigned a j pairing)
                // if (e1.env_ind==4 && proper_i==7) {std::cout<<"MATCH "<<proper_i<<" "<<proper_j<<" "<<std::endl;}
                vec_map.insert(boost::bimap<unsigned int, unsigned int>::value_type(i, j));
                // vec_map.insert(boost::bimap<unsigned int, unsigned int>::value_type(proper_i, proper_j));
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
void MatchEnv::cluster(const vec3<float> *points, unsigned int Np, float threshold)
    {
    assert(points);
    assert(Np > 0);
    assert(threshold > 0);

    // reallocate the m_env_index array for safety
    m_env_index = boost::shared_array<unsigned int>(new unsigned int[Np]);

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
                    // std::cout<<"particle pair: "<<i<<" "<<j<<std::endl;
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
    e0.ignore = true;

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

//! Populate the m_env_index and m_env arrays.
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
        if (dj.s[i].ignore == false)
            {
            unsigned int c = dj.find(i);
            // insert the set into the mapping if we haven't seen it before.
            // also grab the vectors that define the set and insert them into m_env
            if (label_map.count(c) == 0)
                {
                label_map[c] = cur_set;
                boost::shared_array<vec3<float> > vecs = dj.getEnv(c);

                if (reLabel == true) { label_ind = label_map[c]; }
                else { label_ind = c; }

                m_env[label_ind] = vecs;

                cur_set++;
                }

            if (reLabel == true) { label_ind = label_map[c]; }
            else { label_ind = c; }

            // label this particle in m_env_index
            m_env_index[particle_ind] = label_ind;
            particle_ind++;
            }
        }

    // specify the number of cluster environments
    m_num_clusters = cur_set;
    }

}; }; // end namespace freud::match_env;
