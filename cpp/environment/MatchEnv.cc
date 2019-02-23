// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <cstdio>

#include "MatchEnv.h"

namespace freud { namespace environment {

// Constructor for EnvDisjointSet
// Taken partially from Cluster.cc
EnvDisjointSet::EnvDisjointSet(unsigned int Np)
    : rank(std::vector<unsigned int>(Np, 0))
    {
    }

// Merge the two sets that elements a and b belong to.
// Taken partially from Cluster.cc
// The vec_map must be a bimap of PROPERLY ORDERED vector indices where those
// of set a are on the left and those of set b are on the right.
// The rotation must take the set of PROPERLY ROTATED vectors b and rotate
// them to match the set of PROPERLY ROTATED vectors a
void EnvDisjointSet::merge(
        const unsigned int a, const unsigned int b,
        BiMap<unsigned int, unsigned int> vec_map, rotmat3<float> rotation)
    {
    assert(a < s.size() && b < s.size());
    assert(s[a].vecs.size() == s[b].vecs.size());
    assert(vec_map.size() == s[a].vecs.size());

    // if tree heights are equal, merge b to a
    if (rank[s[a].env_ind] == rank[s[b].env_ind])
        {
        // Get the ENTIRE set that corresponds to head_b.
        unsigned int head_b = find(b);
        std::vector<unsigned int> m_set = findSet(head_b);
        for (unsigned int n = 0; n < m_set.size(); n++)
            {
            // Go through the entire tree/set.
            unsigned int node = m_set[n];
            // Make a copy of the old set of vector indices for this
            // particular node.
            std::vector<unsigned int> old_node_vec_ind = s[node].vec_ind;

            // Set the vector indices properly.
            // Take the LEFT MAP view of the proper_a<->proper_b bimap.
            // Iterate over the values of proper_a_ind IN ORDER, find the
            // value of proper_b_ind that corresponds to each proper_a_ind,
            // and set it properly.
            for (unsigned int proper_a_ind=0;
                 proper_a_ind<vec_map.size(); proper_a_ind++)
                {
                unsigned int proper_b_ind = vec_map.left[proper_a_ind];

                // old_node_vec_ind[proper_b_ind] is "relative_b_ind"
                s[node].vec_ind[proper_a_ind] = old_node_vec_ind[proper_b_ind];
                }

            // set the environment index properly
            s[node].env_ind = s[a].env_ind;

            // set the proper orientation. ORDER MATTERS since rotations
            // don't commute in 3D.
            s[node].proper_rot = rotation*s[node].proper_rot;

            // we've added another leaf to the tree or whatever the lingo is.
            rank[s[a].env_ind]++;

            }
        }
    else
        {
        // merge the shorter tree to the taller one
        if (rank[s[a].env_ind] > rank[s[b].env_ind])
            {
            // Get the ENTIRE set that corresponds to head_b.
            unsigned int head_b = find(b);
            std::vector<unsigned int> m_set = findSet(head_b);
            for (unsigned int n = 0; n < m_set.size(); n++)
                {
                // Go through the entire tree/set.
                unsigned int node = m_set[n];
                // Make a copy of the old set of vector indices for this
                // particular node. This is complicated and weird.
                std::vector<unsigned int> old_node_vec_ind = s[node].vec_ind;

                // Set the vector indices properly.
                // Take the LEFT MAP view of the proper_a<->proper_b bimap.
                // Iterate over the values of proper_a_ind IN ORDER, find the
                // value of proper_b_ind that corresponds to each proper_a_ind,
                // and set it properly.
                for (unsigned int proper_a_ind=0;
                     proper_a_ind<vec_map.size(); proper_a_ind++)
                    {
                    unsigned int proper_b_ind = vec_map.left[proper_a_ind];

                    // old_node_vec_ind[proper_b_ind] is "relative_b_ind"
                    s[node].vec_ind[proper_a_ind] = old_node_vec_ind[proper_b_ind];
                    }
                // set the environment index properly
                s[node].env_ind = s[a].env_ind;

                // set the proper orientation. ORDER MATTERS since rotations
                // don't commute in 3D.
                s[node].proper_rot = rotation*s[node].proper_rot;

                // we've added another leaf to the tree or whatever the lingo is.
                rank[s[a].env_ind]++;
                }
            }
        else
            {
            rotmat3<float> rotationT = transpose(rotation);
            // Get the ENTIRE set that corresponds to head_a.
            unsigned int head_a = find(a);
            std::vector<unsigned int> m_set = findSet(head_a);
            for (unsigned int n = 0; n < m_set.size(); n++)
                {
                // Go through the entire tree/set.
                unsigned int node = m_set[n];
                // Make a copy of the old set of vector indices for this
                // particular node. This is complicated and weird.
                std::vector<unsigned int> old_node_vec_ind = s[node].vec_ind;

                // Set the vector indices properly.
                // Take the RIGHT MAP view of the proper_a<->proper_b bimap.
                // Iterate over the values of proper_b_ind IN ORDER, find the
                // value of proper_a_ind that corresponds to each proper_b_ind,
                // and set it properly.
                for (unsigned int proper_b_ind=0;
                     proper_b_ind<vec_map.size(); proper_b_ind++)
                    {
                    unsigned int proper_a_ind = vec_map.right[proper_b_ind];

                    // old_node_vec_ind[proper_a_ind] is "relative_a_ind"
                    s[node].vec_ind[proper_b_ind] = old_node_vec_ind[proper_a_ind];
                    }

                // set the environment index properly
                s[node].env_ind = s[b].env_ind;

                // set the proper orientation. ORDER MATTERS since rotations
                // don't commute in 3D.
                // note that here we are rotating vector set proper_a such
                // that it matches vector set proper_b, so we need to multiply
                // by the INVERSE (transpose) of the matrix rotation.
                s[node].proper_rot = rotationT*s[node].proper_rot;

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
// Values returned: the actual locations of the nodes in s. (i.e. if i is
// returned, the node is accessed by s[i]).
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

// Get the vectors corresponding to environment head index m. Vectors are
// averaged over all members of the environment cluster.
// If environment m doesn't exist as a HEAD in the set, throw an error.
std::shared_ptr<vec3<float> > EnvDisjointSet::getAvgEnv(const unsigned int m)
    {
    assert(s.size() > 0);
    bool invalid_ind = true;

    std::shared_ptr<vec3<float> > env(new vec3<float> [m_max_num_neigh],
                                      std::default_delete<vec3<float>[]>());
    for (unsigned int n = 0; n < m_max_num_neigh; n++)
        {
        env.get()[n] = vec3<float>(0.0,0.0,0.0);
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
                assert(s[i].vec_ind.size() <= m_max_num_neigh);
                assert(s[i].vecs.size() <= m_max_num_neigh);
                assert(s[i].num_vecs == s[m].num_vecs);
                // loop through the vectors, getting them properly indexed
                // add them to env
                for (unsigned int proper_ind = 0;
                     proper_ind < s[i].vecs.size(); proper_ind++)
                    {
                    unsigned int relative_ind = s[i].vec_ind[proper_ind];
                    env.get()[proper_ind] += s[i].proper_rot*s[i].vecs[relative_ind];
                    }
                N += float(1);
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
        // loop through the vectors in env now, dividing by the total number
        // of contributing particle environments to make an average
        for (unsigned int n = 0; n < m_max_num_neigh; n++)
            {
            vec3<float> normed = env.get()[n]/N;
            env.get()[n] = normed;
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
    for (unsigned int n = 0; n < m_max_num_neigh; n++)
        {
        env.push_back(vec3<float>(0.0,0.0,0.0));
        }

    // loop through the vectors, getting them properly indexed
    // add them to env
    for (unsigned int proper_ind = 0; proper_ind < s[m].vecs.size(); proper_ind++)
        {
        unsigned int relative_ind = s[m].vec_ind[proper_ind];
        env[proper_ind] += s[m].proper_rot*s[m].vecs[relative_ind];
        }

    return env;
    }

// Constructor
MatchEnv::MatchEnv(const box::Box& box, float rmax, unsigned int k)
    :m_box(box), m_rmax(rmax), m_k(k)
    {
    m_Np = 0;
    m_num_clusters = 0;
    m_maxk = 0;
    if (m_rmax < 0.0f)
        throw std::invalid_argument("rmax must be positive!");
    m_rmaxsq = m_rmax * m_rmax;
    }

// Destructor
MatchEnv::~MatchEnv()
    {
    }

// Build and return a local environment surrounding a particle.
// Label its environment with env_ind.
Environment MatchEnv::buildEnv(
        const size_t *neighbor_list, size_t num_bonds,
        size_t &bond, const vec3<float> *points,
        unsigned int i, unsigned int env_ind,
        bool hard_r)
    {
    Environment ei = Environment();
    // set the environment index equal to the particle index
    ei.env_ind = env_ind;

    vec3<float> p = points[i];
    for(; bond < num_bonds && neighbor_list[2*bond] == i; ++bond)
            {
            // compute vec{r} between the two particles
            const size_t j(neighbor_list[2*bond + 1]);
            if (i != j)
                {
                vec3<float> delta = m_box.wrap(points[j]-p);
                ei.addVec(delta);
                }
            }

    return ei;
    }

// Is the (PROPERLY REGISTERED) environment e2 similar to the (PROPERLY
// REGISTERED) environment e1?
// If so, return a std::pair of the rotation
// matrix that takes the vectors of e2 to the vectors of e1 AND the
// mapping between the properly indexed vectors of the environments that
// will make them correspond to each other.
// If not, return a std::pair of the identity matrix AND an empty map.
// The threshold is a unitless number, which we multiply by the length scale
// of the MatchEnv instance, rmax.
// This quantity is the maximum squared magnitude of the vector difference
// between two vectors, below which you call them matching.
// The bool registration controls whether we first use brute force registration
// to orient the second set of vectors such that it minimizes the RMSD between
// the two sets.
std::pair<rotmat3<float>, BiMap<unsigned int, unsigned int> > MatchEnv::isSimilar(
        Environment& e1, Environment& e2, float threshold_sq, bool registration)
    {
    BiMap<unsigned int, unsigned int> vec_map;
    rotmat3<float> rotation = rotmat3<float>(); // this initializes to the identity matrix

    // If the vector sets do not have equal numbers of vectors, just return
    // an empty map since the 1-1 bimapping will be too weird in this case.
    if (e1.vecs.size() != e2.vecs.size())
        {
        return std::pair<rotmat3<float>, BiMap<unsigned int, unsigned int> >(rotation, vec_map);
        }

    std::vector< vec3<float> > v1(e1.vecs.size());
    std::vector< vec3<float> > v2(e2.vecs.size());

    // get the vectors into the proper orientation and order with respect to
    // their parent environment
    for (unsigned int m = 0; m < e1.vecs.size(); m++)
        {
        v1[m] = e1.proper_rot*e1.vecs[e1.vec_ind[m]];
        v2[m] = e2.proper_rot*e2.vecs[e2.vec_ind[m]];
        }

    // If we have to register, first find the rotated set of v2 that best maps
    // to v1. The Fit operation CHANGES v2.
    if (registration == true)
        {
        registration::RegisterBruteForce r = registration::RegisterBruteForce(v1);
        r.Fit(v2);
        // get the optimal rotation to take v2 to v1
        std::vector<vec3<float> > rot = r.getRotation();
        // this must be a 3x3 matrix. if it isn't, something has gone wrong.
        assert(rot.size() == 3);
        rotation = rotmat3<float>(rot[0], rot[1], rot[2]);
        BiMap<unsigned int, unsigned int> tmp_vec_map = r.getVecMap();

        for (BiMap<unsigned int, unsigned int>::const_iterator it = tmp_vec_map.begin();
                it != tmp_vec_map.end(); ++it)
            {
            // RegisterBruteForce has found the vector mapping that results in
            // minimal RMSD, as best as it can figure out.
            // Does this vector mapping pass the more stringent criterion
            // imposed by the threshold?
            vec3<float> delta = v1[(*it)->first] - v2[(*it)->second];
            float rsq = dot(delta, delta);
            if (rsq < threshold_sq*m_rmaxsq)
                {
                vec_map.emplace((*it)->first, (*it)->second);
                }
            }
        }

    // if we didn't have to register, compare all combinations of vectors
    else
        {
        for (unsigned int i = 0; i < e1.vecs.size(); i++)
            {
            for (unsigned int j = 0; j < e2.vecs.size(); j++)
                {
                vec3<float> delta = v1[i] - v2[j];
                float rsq = dot(delta, delta);
                if (rsq < threshold_sq*m_rmaxsq)
                    {
                    // these vectors are deemed "matching"
                    // since this is a bimap, this (i,j) pair is only inserted
                    // if j has not already been assigned an i pairing.
                    // (ditto with i not being assigned a j pairing)
                    vec_map.emplace(i, j);
                    }
                }
            }
        }


    // if every vector has been paired with every other vector, return this bimap
    if (vec_map.size() == e1.vecs.size())
        {
        return std::pair<rotmat3<float>, BiMap<unsigned int, unsigned int> >(rotation, vec_map);
        }
    // otherwise, return an empty bimap
    else
        {
        BiMap<unsigned int, unsigned int> empty_map;
        return std::pair<rotmat3<float>, BiMap<unsigned int, unsigned int> >(rotation, empty_map);
        }
    }

// Overload: is the set of vectors refPoints2 similar to the set of vectors refPoints1?
// Construct the environments accordingly, and utilize isSimilar() as above.
// Return a std map for ease of use.
// The bool registration controls whether we first use brute force registration
// to orient the second set of vectors such that it minimizes the RMSD between
// the two sets. If registration=True, then refPoints2 is CHANGED by this function.
std::map<unsigned int, unsigned int> MatchEnv::isSimilar(
        const vec3<float> *refPoints1, vec3<float> *refPoints2,
        unsigned int numRef, float threshold_sq, bool registration)
    {
    assert(refPoints1);
    assert(refPoints2);

    // create the environment characterized by refPoints1. Index it as 0.
    // set the IGNORE flag to true, since this is not an environment we have
    // actually encountered in the simulation.
    Environment e0 = Environment();
    e0.env_ind = 0;
    e0.ghost = true;

    // create the environment characterized by refPoints2. Index it as 1.
    // set the IGNORE flag to true again.
    Environment e1 = Environment();
    e1.env_ind = 1;
    e1.ghost = true;

    // loop through all the vectors in refPoints1 and refPoints2 and add them
    // to the environments.
    // wrap all the vectors back into the box. I think this is necessary since
    // all the vectors that will be added to actual particle environments will
    // be wrapped into the box as well.
    for (unsigned int i = 0; i < numRef; i++)
        {
        vec3<float> p0 = m_box.wrap(refPoints1[i]);
        vec3<float> p1 = m_box.wrap(refPoints2[i]);
        e0.addVec(p0);
        e1.addVec(p1);
        }

    // call isSimilar for e0 and e1
    std::pair<rotmat3<float>, BiMap<unsigned int, unsigned int> > mapping = isSimilar(
            e0, e1, threshold_sq, registration);
    rotmat3<float> rotation = mapping.first;
    BiMap<unsigned int, unsigned int> vec_map = mapping.second;

    // Convert BiMap to a std::map
    std::map<unsigned int, unsigned int> std_vec_map;
    for (BiMap<unsigned int, unsigned int>::const_iterator it = vec_map.begin();
            it != vec_map.end(); ++it)
        {
        std_vec_map[(*it)->first] = (*it)->second;
        }

    // update refPoints2 in case registration has taken place
    for (unsigned int i = 0; i < numRef; i++)
        {
        refPoints2[i] = rotation*e1.vecs[i];
        }

    // return the vector map
    return std_vec_map;
    }

// Get the somewhat-optimal RMSD between the environment e1 and the
// environment e2.
// Return a std::pair of the rotation matrix that takes the vectors of e2 to
// the vectors of e1 AND the mapping between the properly indexed vectors of
// the environments that gives this RMSD.
// Populate the associated minimum RMSD.
// The bool registration controls whether we first use brute force registration
// to orient the second set of vectors such that it minimizes the RMSD between
// the two sets.
// NOTE that this does not guarantee an absolutely minimal RMSD. It doesn't
// figure out the optimal permutation of BOTH sets of vectors to minimize the
// RMSD. Rather, it just figures out the optimal permutation of the second set,
// the vector set used in the argument below.
// To fully solve this, we need to use the Hungarian algorithm or some other
// way of solving the so-called assignment problem.
std::pair<rotmat3<float>, BiMap<unsigned int, unsigned int> > MatchEnv::minimizeRMSD(
        Environment& e1, Environment& e2, float& min_rmsd, bool registration)
    {
    BiMap<unsigned int, unsigned int> vec_map;
    rotmat3<float> rotation = rotmat3<float>(); // this initializes to the identity matrix

    // If the vector sets do not have equal numbers of vectors, force the map
    // to be empty since it can never be 1-1.
    // Return the empty vec_map and the identity matrix, and minRMSD = -1.
    if (e1.vecs.size() != e2.vecs.size())
        {
        min_rmsd = -1.0;
        return std::pair<rotmat3<float>, BiMap<unsigned int, unsigned int> >(rotation, vec_map);
        }

    std::vector< vec3<float> > v1(e1.vecs.size());
    std::vector< vec3<float> > v2(e2.vecs.size());

    // Get the vectors into the proper orientation and order with respect
    // to their parent environment
    for (unsigned int m = 0; m < e1.vecs.size(); m++)
        {
        v1[m] = e1.proper_rot*e1.vecs[e1.vec_ind[m]];
        v2[m] = e2.proper_rot*e2.vecs[e2.vec_ind[m]];
        }

    // call RegisterBruteForce::Fit and update min_rmsd accordingly
    registration::RegisterBruteForce r = registration::RegisterBruteForce(v1);
    // If we have to register, first find the rotated set of v2 that best
    // maps to v1. The Fit operation CHANGES v2.
    if (registration == true)
        {
        r.Fit(v2);
        // get the optimal rotation to take v2 to v1
        std::vector<vec3<float> > rot = r.getRotation();
        // this must be a 3x3 matrix. if it isn't, something has gone wrong.
        assert(rot.size() == 3);
        rotation = rotmat3<float>(rot[0], rot[1], rot[2]);
        min_rmsd = r.getRMSD();
        vec_map = r.getVecMap();
        }
    else
        {
        // this will populate vec_map with the correct mapping
        min_rmsd = r.AlignedRMSDTree(registration::makeEigenMatrix(v2), vec_map);
        }

    // return the rotation matrix and bimap
    return std::pair<rotmat3<float>, BiMap<unsigned int, unsigned int> >(rotation, vec_map);
    }

// Overload: Get the somewhat-optimal RMSD between the set of vectors
// refPoints1 and the set of vectors refPoints2.
// Construct the environments accordingly, and utilize minimizeRMSD() as above.
// Arguments are pointers to interface directly with python.
// Return a std::map (for ease of use) with the mapping between vectors
// refPoints1 and refPoints2 that gives this RMSD.
// Populate the associated minimum RMSD.
// The bool registration controls whether we first use brute force
// registration to orient the second set of vectors such that it minimizes
// the RMSD between the two sets.
// NOTE that this does not guarantee an absolutely minimal RMSD. It doesn't
// figure out the optimal permutation of BOTH sets of vectors to minimize the
// RMSD. Rather, it just figures out the optimal permutation of the second
// set, the vector set used in the argument below.
// To fully solve this, we need to use the Hungarian algorithm or some other
// way of solving the so-called assignment problem.
std::map<unsigned int, unsigned int> MatchEnv::minimizeRMSD(
        const vec3<float> *refPoints1, vec3<float> *refPoints2,
        unsigned int numRef, float& min_rmsd, bool registration)
    {
    assert(refPoints1);
    assert(refPoints2);

    // create the environment characterized by refPoints1. Index it as 0.
    // set the IGNORE flag to true, since this is not an environment we have
    // actually encountered in the simulation.
    Environment e0 = Environment();
    e0.env_ind = 0;
    e0.ghost = true;

    // create the environment characterized by refPoints2. Index it as 1.
    // set the IGNORE flag to true again.
    Environment e1 = Environment();
    e1.env_ind = 1;
    e1.ghost = true;

    // loop through all the vectors in refPoints1 and refPoints2 and add them
    // to the environments.
    // wrap all the vectors back into the box. I think this is necessary since
    // all the vectors that will be added to actual particle environments will
    // be wrapped into the box as well.
    for (unsigned int i = 0; i < numRef; i++)
        {
        vec3<float> p0 = m_box.wrap(refPoints1[i]);
        vec3<float> p1 = m_box.wrap(refPoints2[i]);
        e0.addVec(p0);
        e1.addVec(p1);
        }

    // call minimizeRMSD for e0 and e1
    float tmp_min_rmsd = -1.0;
    std::pair<rotmat3<float>, BiMap<unsigned int, unsigned int> > mapping = minimizeRMSD(
            e0, e1, tmp_min_rmsd, registration);
    rotmat3<float> rotation = mapping.first;
    BiMap<unsigned int, unsigned int> vec_map = mapping.second;
    min_rmsd = tmp_min_rmsd;

    // Convert BiMap to a std::map
    std::map<unsigned int, unsigned int> std_vec_map;
    for (BiMap<unsigned int, unsigned int>::const_iterator it = vec_map.begin();
            it != vec_map.end(); ++it)
        {
        std_vec_map[(*it)->first] = (*it)->second;
        }

    // update refPoints2 in case registration has taken place
    for (unsigned int i = 0; i < numRef; i++)
        {
        refPoints2[i] = rotation*e1.vecs[i];
        }

    // return the vector map
    return std_vec_map;
    }

// Determine clusters of particles with matching environments
// This is taken from Cluster.cc and SolLiq.cc and LocalQlNear.cc
void MatchEnv::cluster(
        const freud::locality::NeighborList *env_nlist,
        const freud::locality::NeighborList *nlist,
        const vec3<float> *points, unsigned int Np, float threshold,
        bool hard_r, bool registration, bool global)
    {
    assert(points);
    assert(Np > 0);
    assert(threshold > 0);

    // reallocate the m_env_index array for safety
    m_env_index = std::shared_ptr<unsigned int>(new unsigned int[Np], std::default_delete<unsigned int[]>());

    m_Np = Np;
    float m_threshold_sq = threshold*threshold;

    nlist->validate(Np, Np);
    const size_t *neighbor_list(nlist->getNeighbors());

    env_nlist->validate(Np, Np);
    const size_t *env_neighbor_list(env_nlist->getNeighbors());
    size_t env_bond(0);
    const size_t env_num_bonds(env_nlist->getNumBonds());

    // create a disjoint set where all particles belong in their own cluster
    EnvDisjointSet dj(m_Np);

    // add all the environments to the set
    // take care, here: set things up s.t. the env_ind of every environment
    // matches its location in the disjoint set.
    // if you don't do this, things will get screwy.
    for (unsigned int i = 0; i < m_Np; i++)
        {
        Environment ei = buildEnv(env_neighbor_list, env_num_bonds, env_bond, points, i, i, hard_r);
        dj.s.push_back(ei);
        m_maxk = std::max(m_maxk, ei.num_vecs);
        dj.m_max_num_neigh = m_maxk;
        }

    // reallocate the m_tot_env array
    unsigned int array_size = Np*m_maxk;
    m_tot_env = std::shared_ptr<vec3<float> >(new vec3<float>[array_size], std::default_delete<vec3<float>[]>());

    size_t bond(0);
    // loop through points
    for (unsigned int i = 0; i < m_Np; i++)
        {
        if (global == false)
            {
            // loop over the neighbors
            for(; bond < nlist->getNumBonds() && neighbor_list[2*bond] == i; ++bond)
                {
                const size_t j(neighbor_list[2*bond + 1]);
                if (i != j)
                    {
                    std::pair<rotmat3<float>, BiMap<unsigned int, unsigned int> > mapping = isSimilar(
                            dj.s[i], dj.s[j], m_threshold_sq, registration);
                    rotmat3<float> rotation = mapping.first;
                    BiMap<unsigned int, unsigned int> vec_map = mapping.second;
                    // if the mapping between the vectors of the environments
                    // is NOT empty, then the environments are similar, so
                    // merge them.
                    if (!vec_map.empty())
                        {
                        // merge the two sets using the disjoint set
                        unsigned int a = dj.find(i);
                        unsigned int b = dj.find(j);
                        if (a != b)
                            dj.merge(i,j,vec_map,rotation);
                        }
                    }
                }
            }
        else
            {
            // loop over all other particles
            for (unsigned int j = i+1; j < m_Np; j++)
                {
                std::pair<rotmat3<float>, BiMap<unsigned int, unsigned int> > mapping = isSimilar(
                        dj.s[i], dj.s[j], m_threshold_sq, registration);
                rotmat3<float> rotation = mapping.first;
                BiMap<unsigned int, unsigned int> vec_map = mapping.second;
                // if the mapping between the vectors of the environments
                // is NOT empty, then the environments are similar, so
                // merge them.
                if (!vec_map.empty())
                    {
                    // merge the two sets using the disjoint set
                    unsigned int a = dj.find(i);
                    unsigned int b = dj.find(j);
                    if (a != b)
                        dj.merge(i,j,vec_map,rotation);
                    }
                }
            }
        }

    // done looping over points. All clusters are now determined. Renumber
    // them from zero to num_clusters-1.
    populateEnv(dj, true);
    }

//! Determine whether particles match a given input motif, characterized by
//  refPoints (of which there are numRef)
void MatchEnv::matchMotif(
        const freud::locality::NeighborList *nlist,
        const vec3<float> *points, unsigned int Np,
        const vec3<float> *refPoints, unsigned int numRef,
        float threshold, bool registration)
    {
    assert(points);
    assert(refPoints);
    assert(numRef == m_k);
    assert(Np > 0);
    assert(threshold > 0);

    // reallocate the m_env_index array for safety
    m_env_index = std::shared_ptr<unsigned int>(new unsigned int[Np], std::default_delete<unsigned int[]>());

    m_Np = Np;
    float m_threshold_sq = threshold*threshold;

    nlist->validate(Np, Np);
    const size_t *neighbor_list(nlist->getNeighbors());

    // create a disjoint set where all particles belong in their own cluster.
    // this has to have ONE MORE environment than there are actual particles,
    // because we're inserting the motif into it.
    EnvDisjointSet dj(m_Np+1);
    dj.m_max_num_neigh = m_k;
    m_maxk = m_k;

    // reallocate the m_tot_env array
    unsigned int array_size = Np*m_maxk;
    m_tot_env = std::shared_ptr<vec3<float> >(new vec3<float>[array_size], std::default_delete<vec3<float>[]>());

    // create the environment characterized by refPoints. Index it as 0.
    // set the IGNORE flag to true, since this is not an environment we have
    // actually encountered in the simulation.
    Environment e0 = Environment();
    e0.env_ind = 0;
    e0.ghost = true;

    // loop through all the vectors in refPoints and add them to the environment.
    // wrap all the vectors back into the box. I think this is necessary since
    // all the vectors that will be added to actual particle environments will
    // be wrapped into the box as well.
    for (unsigned int i = 0; i < numRef; i++)
        {
        vec3<float> p = m_box.wrap(refPoints[i]);
        e0.addVec(p);
        }

    // add this environment to the set
    dj.s.push_back(e0);

    size_t bond(0);
    const size_t num_bonds(nlist->getNumBonds());

    // loop through the particles and add their environments to the set
    // take care, here: set things up s.t. the env_ind of every environment
    // matches its location in the disjoint set.
    // if you don't do this, things will get screwy.
    for (unsigned int i = 0; i < m_Np; i++)
        {
        unsigned int dummy = i+1;
        Environment ei = buildEnv(neighbor_list, num_bonds, bond, points, i, dummy, false);
        dj.s.push_back(ei);

        // if the environment matches e0, merge it into the e0 environment set
        std::pair<rotmat3<float>, BiMap<unsigned int, unsigned int> > mapping = isSimilar(
                dj.s[0], dj.s[dummy], m_threshold_sq, registration);
        rotmat3<float> rotation = mapping.first;
        BiMap<unsigned int, unsigned int> vec_map = mapping.second;
        // if the mapping between the vectors of the environments is NOT empty,
        // then the environments are similar.
        if (!vec_map.empty())
            {
            dj.merge(0, dummy, vec_map, rotation);
            }
        }

    // DON'T renumber the clusters in the disjoint set from zero to
    // num_clusters-1. The way I have set it up here, the "0th" cluster
    // is the one that matches the motif.
    populateEnv(dj, false);

    }

//! Rotate (if registration=True) and permute the environments of all particles
//  to minimize their RMSD wrt a given input motif, characterized by refPoints
//  (of which there are numRef).
//  Returns a vector of minimal RMSD values, one value per particle.
//  NOTE that this does not guarantee an absolutely minimal RMSD. It doesn't
//  figure out the optimal permutation of BOTH sets of vectors to minimize the
//  RMSD. Rather, it just figures out the optimal permutation of the second
//  set, the vector set used in the argument below.
//  To fully solve this, we need to use the Hungarian algorithm or some other
//  way of solving the so-called assignment problem.
std::vector<float> MatchEnv::minRMSDMotif(
        const freud::locality::NeighborList *nlist,
        const vec3<float> *points, unsigned int Np,
        const vec3<float> *refPoints, unsigned int numRef,
        bool registration)
    {
    assert(points);
    assert(refPoints);
    assert(numRef == m_k);
    assert(Np > 0);

    // reallocate the m_env_index array for safety
    m_env_index = std::shared_ptr<unsigned int>(new unsigned int[Np], std::default_delete<unsigned int[]>());

    m_Np = Np;
    std::vector<float> min_rmsd_vec(m_Np);

    nlist->validate(Np, Np);
    const size_t *neighbor_list(nlist->getNeighbors());

    // create a disjoint set where all particles belong in their own cluster.
    // this has to have ONE MORE environment than there are actual particles,
    // because we're inserting the motif into it.
    EnvDisjointSet dj(m_Np+1);
    dj.m_max_num_neigh = m_k;
    m_maxk = m_k;

    // reallocate the m_tot_env array
    unsigned int array_size = Np*m_maxk;
    m_tot_env = std::shared_ptr<vec3<float> >(new vec3<float>[array_size], std::default_delete<vec3<float>[]>());

    // create the environment characterized by refPoints. Index it as 0.
    // set the IGNORE flag to true, since this is not an environment we
    // have actually encountered in the simulation.
    Environment e0 = Environment();
    e0.env_ind = 0;
    e0.ghost = true;

    // loop through all the vectors in refPoints and add them to the environment.
    // wrap all the vectors back into the box. I think this is necessary since
    // all the vectors that will be added to actual particle environments will
    // be wrapped into the box as well.
    for (unsigned int i = 0; i < numRef; i++)
        {
        vec3<float> p = m_box.wrap(refPoints[i]);
        e0.addVec(p);
        }

    // add this environment to the set
    dj.s.push_back(e0);

    size_t bond(0);
    const size_t num_bonds(nlist->getNumBonds());

    // loop through the particles and add their environments to the set
    // take care, here: set things up s.t. the env_ind of every environment
    // matches its location in the disjoint set.
    // if you don't do this, things will get screwy.
    for (unsigned int i = 0; i < m_Np; i++)
        {
        unsigned int dummy = i+1;
        Environment ei = buildEnv(neighbor_list, num_bonds, bond, points, i, dummy, false);
        dj.s.push_back(ei);

        // if the environment matches e0, merge it into the e0 environment set
        float min_rmsd = -1.0;
        std::pair<rotmat3<float>, BiMap<unsigned int, unsigned int> > mapping = minimizeRMSD(
                dj.s[0], dj.s[dummy], min_rmsd, registration);
        rotmat3<float> rotation = mapping.first;
        BiMap<unsigned int, unsigned int> vec_map = mapping.second;
        // populate the min_rmsd vector
        min_rmsd_vec[i] = min_rmsd;

        // if the mapping between the vectors of the environments is NOT
        // empty, then the environments are similar.
        // minimizeRMSD should always return a non-empty vec_map, except if
        // e0 and e1 have different numbers of vectors.
        if (!vec_map.empty())
            {
            dj.merge(0, dummy, vec_map, rotation);
            }
        }

    // DON'T renumber the clusters in the disjoint set from zero to
    // num_clusters-1. The way I have set it up here, the "0th" cluster
    // is the one that matches the motif.
    populateEnv(dj, false);

    return min_rmsd_vec;
    }

//! Populate the m_env_index, m_env and m_tot_env arrays.
//! Renumber the clusters in the disjoint set dj from zero to num_clusters-1,
//  if that is called.
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
                std::shared_ptr<vec3<float> > vecs = dj.getAvgEnv(c);

                if (reLabel == true) { label_ind = label_map[c]; }
                else { label_ind = c; }

                m_env[label_ind] = vecs;

                cur_set++;
                }

            if (reLabel == true) { label_ind = label_map[c]; }
            else { label_ind = c; }

            // label this particle in m_env_index
            m_env_index.get()[particle_ind] = label_ind;
            // add the particle environment to m_tot_env
            // get a pointer to the start of m_tot_env
            vec3<float> *start = m_tot_env.get();
            // loop through part_vecs and add them
            for (unsigned int m = 0; m < part_vecs.size(); m++)
                {
                unsigned int index = particle_ind*m_maxk + m;
                start[index] = part_vecs[m];
                }
            particle_ind++;
            }
        }

    // specify the number of cluster environments
    m_num_clusters = cur_set;
    }

}; }; // end namespace freud::environment
