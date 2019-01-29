// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <stdexcept>
#include <tbb/tbb.h>
#include <tuple>

#include "AABBQuery.h"
#include "LinkCell.h"

namespace freud { namespace locality {

AABBQuery::AABBQuery(): m_rcut(0)
    {
    }

AABBQuery::~AABBQuery()
    {
    }

void AABBQuery::compute(box::Box& box, float rcut,
        const vec3<float> *ref_points, unsigned int Nref,
        const vec3<float> *points, unsigned int Np,
        bool exclude_ii)
    {
    m_box = box;
    m_rcut = rcut;
    m_Ntotal = Nref + Np;

    // Allocate memory and create image vectors
    setupTree(Np);

    // Build the tree
    buildTree(points, Np);

    // Now walk the tree
    traverseTree(ref_points, Nref, points, Np, exclude_ii);
    }

void AABBQuery::setupTree(unsigned int Np)
    {
    m_aabbs.resize(Np);
    updateImageVectors();
    }

void AABBQuery::updateImageVectors()
    {
    vec3<float> nearest_plane_distance = m_box.getNearestPlaneDistance();
    vec3<bool> periodic = m_box.getPeriodic();
    float rmax = m_rcut;
    if ((periodic.x && nearest_plane_distance.x < rmax * 2.0) ||
        (periodic.y && nearest_plane_distance.y < rmax * 2.0) ||
        (!m_box.is2D() && periodic.z && nearest_plane_distance.z < rmax * 2.0))
        {
        throw std::runtime_error("The AABBQuery rcut is too large for this box.");
        }

    // Now compute the image vectors
    // Each dimension increases by one power of 3
    unsigned int n_dim_periodic = (unsigned int)(periodic.x + periodic.y + (!m_box.is2D())*periodic.z);
    m_n_images = 1;
    for (unsigned int dim = 0; dim < n_dim_periodic; ++dim)
        {
        m_n_images *= 3;
        }

    // Reallocate memory if necessary
    if (m_n_images > m_image_list.size())
        {
        m_image_list.resize(m_n_images);
        }

    vec3<float> latt_a = vec3<float>(m_box.getLatticeVector(0));
    vec3<float> latt_b = vec3<float>(m_box.getLatticeVector(1));
    vec3<float> latt_c = vec3<float>(0.0, 0.0, 0.0);
    if (!m_box.is2D())
        {
        latt_c = vec3<float>(m_box.getLatticeVector(2));
        }

    // There is always at least 1 image, which we put as our first thing to look at
    m_image_list[0] = vec3<float>(0.0, 0.0, 0.0);

    // Iterate over all other combinations of images
    unsigned int n_images = 1;
    for (int i=-1; i <= 1 && n_images < m_n_images; ++i)
        {
        for (int j=-1; j <= 1 && n_images < m_n_images; ++j)
            {
            for (int k=-1; k <= 1 && n_images < m_n_images; ++k)
                {
                if (!(i == 0 && j == 0 && k == 0))
                    {
                    // Skip any periodic images if we don't have periodicity
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

void AABBQuery::buildTree(const vec3<float> *points, unsigned int Np)
    {
    // Construct a point AABB for each point
    for (unsigned int i = 0; i < Np; ++i)
        {
        // Make a point AABB
        vec3<float> my_pos(points[i]);
        if (m_box.is2D())
            my_pos.z = 0;
        m_aabbs[i] = AABB(my_pos, i);
        }

    // Call the tree build routine, one tree per type
    m_aabb_tree.buildTree(m_aabbs.data(), Np);
    }

void AABBQuery::traverseTree(const vec3<float> *ref_points, unsigned int Nref,
        const vec3<float> *points, unsigned int Np, bool exclude_ii)
    {
    if (!Np)
        return;

    float r_cutsq = m_rcut * m_rcut;

    typedef std::vector<std::tuple<size_t, size_t, float> > BondVector;
    typedef std::vector<BondVector> BondVectorVector;
    typedef tbb::enumerable_thread_specific<BondVectorVector> ThreadBondVector;
    ThreadBondVector bond_vectors;

    // Loop over all reference points in parallel
    parallel_for(tbb::blocked_range<size_t>(0, Nref),
        [=, &bond_vectors] (const tbb::blocked_range<size_t> &r)
        {
        ThreadBondVector::reference bond_vector_vectors(bond_vectors.local());
        bond_vector_vectors.emplace_back();
        BondVector &bond_vector(bond_vector_vectors.back());

        // Loop over this thread's reference points
        for (size_t i(r.begin()); i != r.end(); ++i)
            {
            // Read in the position of i
            vec3<float> pos_i(ref_points[i]);
            if (m_box.is2D())
                {
                pos_i.z = 0;
                }

            // Loop over image vectors
            for (unsigned int cur_image = 0; cur_image < m_n_images; ++cur_image)
                {
                // Make an AABB for the image of this point
                vec3<float> pos_i_image = pos_i + m_image_list[cur_image];
                AABBSphere asphere = AABBSphere(pos_i_image, m_rcut);

                // Stackless traversal of the tree
                for (unsigned int cur_node_idx = 0;
                     cur_node_idx < m_aabb_tree.getNumNodes();
                     ++cur_node_idx)
                    {
                    if (overlap(m_aabb_tree.getNodeAABB(cur_node_idx), asphere))
                        {
                        if (m_aabb_tree.isNodeLeaf(cur_node_idx))
                            {
                            for (unsigned int cur_p = 0;
                                 cur_p < m_aabb_tree.getNodeNumParticles(cur_node_idx);
                                 ++cur_p)
                                {
                                // Neighbor j
                                const unsigned int j = m_aabb_tree.getNodeParticleTag(cur_node_idx, cur_p);

                                // Determine whether to skip self-interaction
                                if (exclude_ii && i == j)
                                    continue;

                                // Read in the position of j
                                vec3<float> pos_j(points[j]);
                                if (m_box.is2D())
                                    {
                                    pos_j.z = 0;
                                    }

                                // Compute distance
                                const vec3<float> drij = pos_j - pos_i_image;
                                const float dr_sq = dot(drij, drij);

                                if (dr_sq < r_cutsq)
                                    {
                                    bond_vector.emplace_back(i, j, 1);
                                    }
                                }
                            }
                        }
                    else
                        {
                        // Skip ahead
                        cur_node_idx += m_aabb_tree.getNodeSkip(cur_node_idx);
                        }
                    } // end stackless search
                } // end loop over images
            } // end loop over reference points
        });

    // Sort neighbors by particle i index
    tbb::flattened2d<ThreadBondVector> flat_bond_vector_groups = tbb::flatten2d(bond_vectors);
    BondVectorVector bond_vector_groups(flat_bond_vector_groups.begin(), flat_bond_vector_groups.end());
    tbb::parallel_sort(bond_vector_groups.begin(), bond_vector_groups.end(), compareFirstNeighborPairs);

    unsigned int num_bonds(0);
    for(BondVectorVector::const_iterator iter(bond_vector_groups.begin());
        iter != bond_vector_groups.end(); ++iter)
        num_bonds += iter->size();

    m_neighbor_list.resize(num_bonds);
    m_neighbor_list.setNumBonds(num_bonds, Nref, Np);

    size_t *neighbor_array(m_neighbor_list.getNeighbors());
    float *neighbor_weights(m_neighbor_list.getWeights());

    // Build nlist structure
    parallel_for(tbb::blocked_range<size_t>(0, bond_vector_groups.size()),
        [=, &bond_vector_groups] (const tbb::blocked_range<size_t> &r)
        {
        size_t bond(0);
        for (size_t group(0); group < r.begin(); ++group)
            bond += bond_vector_groups[group].size();

        for (size_t group(r.begin()); group < r.end(); ++group)
            {
            const BondVector &vec(bond_vector_groups[group]);
            for (BondVector::const_iterator iter(vec.begin());
                iter != vec.end(); ++iter, ++bond)
                {
                std::tie(neighbor_array[2*bond], neighbor_array[2*bond + 1],
                    neighbor_weights[bond]) = *iter;
                }
            }
        });
    }

}; }; // end namespace freud::locality
