// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <stdexcept>
#include "AABBQuery.h"

namespace freud { namespace locality {

AABBQuery::AABBQuery(): m_rcut(1.0)
    {
    }

AABBQuery::~AABBQuery()
    {
    }

void AABBQuery::compute(box::Box& box,
        const vec3<float> *ref_points, unsigned int Nref,
        const vec3<float> *points, unsigned int Np,
        bool exclude_ii)
    {
    m_box = box;
    m_Ntotal = Nref + Np;
    m_num_per_type.clear();
    m_num_per_type.push_back(Nref);
    m_num_per_type.push_back(Np);

    // TODO: Do particles need to be wrapped?

    // allocate the memory as needed and sort particles
    setupTree();

    // build the trees
    buildTree(ref_points, Nref, points, Np);

    // now walk the trees
    traverseTree(ref_points, Nref, points, Np, exclude_ii);
    }

void AABBQuery::setupTree()
    {
    m_aabbs.resize(m_Ntotal);
    m_map_pid_tree.resize(m_Ntotal);
    m_aabb_trees.resize(2);
    m_type_head.resize(2, 0);
    mapParticlesByType();
    updateImageVectors();
    }

void AABBQuery::mapParticlesByType()
    {
    unsigned int ref_point_counter = 0;
    unsigned int point_counter = 0;
    for (unsigned int i=0; i < m_Ntotal; ++i)
        {
        if (i < m_num_per_type[0])
            {
            m_map_pid_tree[i] = ref_point_counter;
            ref_point_counter++;
            }
        else
            {
            m_map_pid_tree[i] = point_counter;
            point_counter++;
            }
        }
    m_type_head[0] = 0;
    m_type_head[1] = m_num_per_type[0];
    }

void AABBQuery::updateImageVectors()
    {
    vec3<float> nearest_plane_distance = m_box.getNearestPlaneDistance();
    vec3<bool> periodic = m_box.getPeriodic();
    float rmax = m_rcut;
    if ((periodic.x && nearest_plane_distance.x <= rmax * 2.0) ||
        (periodic.y && nearest_plane_distance.y <= rmax * 2.0) ||
        (!m_box.is2D() && periodic.z && nearest_plane_distance.z <= rmax * 2.0))
        {
        throw std::runtime_error("The AABBQuery rcut is too large for this box.");
        }

    // now compute the image vectors
    // each dimension increases by one power of 3
    unsigned int n_dim_periodic = (unsigned int)(periodic.x + periodic.y + (!m_box.is2D())*periodic.z);
    m_n_images = 1;
    for (unsigned int dim = 0; dim < n_dim_periodic; ++dim)
        {
        m_n_images *= 3;
        }

    // reallocate memory if necessary
    if (m_n_images > m_image_list.size())
        {
        m_image_list.resize(m_n_images);
        }

    vec3<float> latt_a = vec3<float>(m_box.getLatticeVector(0));
    vec3<float> latt_b = vec3<float>(m_box.getLatticeVector(1));
    vec3<float> latt_c = vec3<float>(m_box.getLatticeVector(2));

    // there is always at least 1 image, which we put as our first thing to look at
    m_image_list[0] = vec3<float>(0.0, 0.0, 0.0);

    // iterate over all other combinations of images, skipping those that are
    unsigned int n_images = 1;
    for (int i=-1; i <= 1 && n_images < m_n_images; ++i)
        {
        for (int j=-1; j <= 1 && n_images < m_n_images; ++j)
            {
            for (int k=-1; k <= 1 && n_images < m_n_images; ++k)
                {
                if (!(i == 0 && j == 0 && k == 0))
                    {
                    // skip any periodic images if we don't have periodicity
                    if (i != 0 && !periodic.x) continue;
                    if (j != 0 && !periodic.y) continue;
                    if (k != 0 && (m_box.is2D() || !periodic.z)) continue;

                    m_image_list[n_images] = float(i) * latt_a + float(j) * latt_b + float(k) * latt_c;
                    ++n_images;
                    }
                }
            }
        }
    }

void AABBQuery::buildTree(const vec3<float> *ref_points, unsigned int Nref,
        const vec3<float> *points, unsigned int Np)
    {
    // construct a point AABB for each particle owned by this rank, and push it
    // into the right spot in the AABB list
    for (unsigned int i=0; i < Nref; ++i)
        {
        // make a ref_point AABB
        const vec3<float> my_pos(ref_points[i]);

        unsigned int my_aabb_idx = m_map_pid_tree[i];
        m_aabbs[my_aabb_idx] = AABB(my_pos, i);
        }
    for (unsigned int i=0; i < Np; ++i)
        {
        // make a point AABB
        const vec3<float> my_pos(points[i]);

        unsigned int my_aabb_idx = m_map_pid_tree[Nref + i];
        m_aabbs[my_aabb_idx] = AABB(my_pos, Nref + i);
        }


    // call the tree build routine, one tree per type
    for (unsigned int i=0; i < m_num_per_type.size(); ++i)
        {
        if (m_num_per_type[i] > 0)
            {
            m_aabb_trees[i].buildTree(m_aabbs.data() + m_type_head[i], m_num_per_type[i]);
            }
        }

    }

void AABBQuery::traverseTree(const vec3<float> *ref_points, unsigned int Nref,
        const vec3<float> *points, unsigned int Np, bool exclude_ii)
    {
    if (!Np)
        return;

    // Loop over all particles
    for (unsigned int i=0; i < Nref; ++i)
        {
        // Read in the current position
        const vec3<float> pos_i = ref_points[i];

        float r_cutsq = m_rcut * m_rcut;

        AABBTree *cur_aabb_tree = &m_aabb_trees[1];

        for (unsigned int cur_image = 0; cur_image < m_n_images; ++cur_image) // for each image vector
            {
            // make an AABB for the image of this particle
            vec3<float> pos_i_image = pos_i + m_image_list[cur_image];
            AABB aabb = AABB(pos_i_image, m_rcut);

            // stackless traversal of the tree
            for (unsigned int cur_node_idx = 0;
                 cur_node_idx < cur_aabb_tree->getNumNodes();
                 ++cur_node_idx)
                {
                if (overlap(cur_aabb_tree->getNodeAABB(cur_node_idx), aabb))
                    {
                    if (cur_aabb_tree->isNodeLeaf(cur_node_idx))
                        {
                        for (unsigned int cur_p = 0; cur_p < cur_aabb_tree->getNodeNumParticles(cur_node_idx); ++cur_p)
                            {
                            // neighbor j
                            unsigned int j = cur_aabb_tree->getNodeParticleTag(cur_node_idx, cur_p);

                            // skip self-interaction
                            bool excluded = (i == j) && exclude_ii;

                            if (!excluded)
                                {
                                // compute distance
                                const vec3<float> pos_j = points[j];
                                const vec3<float> drij = pos_j - pos_i_image;
                                float dr_sq = dot(drij, drij);

                                if (dr_sq <= r_cutsq)
                                    {
                                    // TODO: Add neighborlist entry
                                    }
                                }
                            }
                        }
                    }
                else
                    {
                    // skip ahead
                    cur_node_idx += cur_aabb_tree->getNodeSkip(cur_node_idx);
                    }
                } // end stackless search
            } // end loop over images
        } // end loop over particles
    }

}; }; // end namespace freud::locality
