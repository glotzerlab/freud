// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <tbb/tbb.h>
#include <tuple>

#include "AABBQuery.h"

namespace freud { namespace locality {

AABBQuery::AABBQuery(const box::Box& box, const vec3<float>* ref_points, unsigned int Nref)
    : NeighborQuery(box, ref_points, Nref)
{
    // Allocate memory and create image vectors
    setupTree(m_Nref);

    // Build the tree
    buildTree(m_ref_points, m_Nref);
}

AABBQuery::~AABBQuery() {}

std::shared_ptr<NeighborQueryIterator> AABBQuery::query(const vec3<float>* points, unsigned int N,
                                                        unsigned int k, float r, float scale,
                                                        bool exclude_ii) const
{
    return std::make_shared<AABBQueryIterator>(this, points, N, k, r, scale, exclude_ii);
}

std::shared_ptr<NeighborQueryIterator> AABBQuery::queryBall(const vec3<float>* points, unsigned int N,
                                                            float r, bool exclude_ii) const
{
    return std::make_shared<AABBQueryBallIterator>(this, points, N, r, exclude_ii);
}

std::shared_ptr<NeighborQueryIterator>
AABBQuery::queryBallUnbounded(const vec3<float>* points, unsigned int N, float r, bool exclude_ii) const
{
    return std::make_shared<AABBQueryBallIterator>(this, points, N, r, exclude_ii, false);
}

void AABBQuery::setupTree(unsigned int Np)
{
    m_aabbs.resize(Np);
}

void AABBQuery::buildTree(const vec3<float>* points, unsigned int Np)
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

void AABBIterator::updateImageVectors(float rmax, bool _check_rmax)
{
    box::Box box = m_neighbor_query->getBox();
    vec3<float> nearest_plane_distance = box.getNearestPlaneDistance();
    vec3<bool> periodic = box.getPeriodic();
    if (_check_rmax)
    {
        if ((periodic.x && nearest_plane_distance.x <= rmax * 2.0)
            || (periodic.y && nearest_plane_distance.y <= rmax * 2.0)
            || (!box.is2D() && periodic.z && nearest_plane_distance.z <= rmax * 2.0))
        {
            throw std::runtime_error("The AABBQuery rcut is too large for this box.");
        }
    }

    // Now compute the image vectors
    // Each dimension increases by one power of 3
    unsigned int n_dim_periodic = (unsigned int) (periodic.x + periodic.y + (!box.is2D()) * periodic.z);
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

    vec3<float> latt_a = vec3<float>(box.getLatticeVector(0));
    vec3<float> latt_b = vec3<float>(box.getLatticeVector(1));
    vec3<float> latt_c = vec3<float>(0.0, 0.0, 0.0);
    if (!box.is2D())
    {
        latt_c = vec3<float>(box.getLatticeVector(2));
    }

    // There is always at least 1 image, which we put as our first thing to look at
    m_image_list[0] = vec3<float>(0.0, 0.0, 0.0);

    // Iterate over all other combinations of images
    unsigned int n_images = 1;
    for (int i = -1; i <= 1 && n_images < m_n_images; ++i)
    {
        for (int j = -1; j <= 1 && n_images < m_n_images; ++j)
        {
            for (int k = -1; k <= 1 && n_images < m_n_images; ++k)
            {
                if (!(i == 0 && j == 0 && k == 0))
                {
                    // Skip any periodic images if we don't have periodicity
                    if (i != 0 && !periodic.x)
                        continue;
                    if (j != 0 && !periodic.y)
                        continue;
                    if (k != 0 && (box.is2D() || !periodic.z))
                        continue;

                    m_image_list[n_images] = float(i) * latt_a + float(j) * latt_b + float(k) * latt_c;
                    ++n_images;
                }
            }
        }
    }
}

NeighborPoint AABBQueryBallIterator::next()
{
    float r_cutsq = m_r * m_r;

    while (cur_p < m_N)
    {
        // Read in the position of current point
        vec3<float> pos_i(m_points[cur_p]);
        if (m_neighbor_query->getBox().is2D())
        {
            pos_i.z = 0;
        }

        // Loop over image vectors
        while (cur_image < m_n_images)
        {
            // Make an AABB for the image of this point
            vec3<float> pos_i_image = pos_i + m_image_list[cur_image];
            AABBSphere asphere = AABBSphere(pos_i_image, m_r);

            // Stackless traversal of the tree
            while (cur_node_idx < m_aabb_query->m_aabb_tree.getNumNodes())
            {
                if (overlap(m_aabb_query->m_aabb_tree.getNodeAABB(cur_node_idx), asphere))
                {
                    if (m_aabb_query->m_aabb_tree.isNodeLeaf(cur_node_idx))
                    {
                        while (cur_ref_p < m_aabb_query->m_aabb_tree.getNodeNumParticles(cur_node_idx))
                        {
                            // Neighbor j
                            const unsigned int j
                                = m_aabb_query->m_aabb_tree.getNodeParticleTag(cur_node_idx, cur_ref_p);

                            // Read in the position of j
                            vec3<float> pos_j((*m_neighbor_query)[j]);
                            if (m_neighbor_query->getBox().is2D())
                            {
                                pos_j.z = 0;
                            }

                            // Compute distance
                            const vec3<float> drij = pos_j - pos_i_image;
                            const float dr_sq = dot(drij, drij);

                            // Increment before possible return.
                            cur_ref_p++;
                            // Check ii exclusion before including the pair.
                            if (dr_sq < r_cutsq && (!m_exclude_ii || cur_p != j))
                            {
                                return NeighborPoint(cur_p, j, sqrt(dr_sq));
                            }
                        }
                    }
                }
                else
                {
                    // Skip ahead
                    cur_node_idx += m_aabb_query->m_aabb_tree.getNodeSkip(cur_node_idx);
                }
                cur_node_idx++;
                cur_ref_p = 0;
            } // end stackless search
            cur_image++;
            cur_node_idx = 0;
        } // end loop over images
        cur_p++;
        cur_image = 0;
    }

    m_finished = true;
    return NeighborQueryIterator::ITERATOR_TERMINATOR;
}

std::shared_ptr<NeighborQueryIterator> AABBQueryBallIterator::query(unsigned int idx)
{
    return this->m_aabb_query->queryBall(&m_points[idx], 1, m_r);
}

NeighborPoint AABBQueryIterator::next()
{
    vec3<float> plane_distance = m_neighbor_query->getBox().getNearestPlaneDistance();
    float min_plane_distance = std::min(plane_distance.x, plane_distance.y);
    float max_plane_distance = std::max(plane_distance.x, plane_distance.y);
    if (!m_neighbor_query->getBox().is2D())
    {
        min_plane_distance = std::min(min_plane_distance, plane_distance.z);
        max_plane_distance = std::max(max_plane_distance, plane_distance.z);
    }

    while (cur_p < m_N)
    {
        // Only try to add new neighbors if there are no neighbors currently cached to return.
        if (!m_current_neighbors.size())
        {
            // Continually perform ball queries until the termination conditions are met.
            while (true)
            {
                // Perform a ball query to get neighbors. Since we are doing
                // this on a per-point basis, we don't pass the exclude_ii
                // parameter through because the indexes won't match. Instead,
                // we have to filter the ii matches after the fact. We also
                // need to do some extra magic to ensure that we allow ball
                // queries to exceed their normal boundaries, which requires
                // the cast performed below to expose the appropriate method to
                // the compiler.
                m_current_neighbors.clear();
                std::shared_ptr<NeighborQueryIterator> ball_it
                    = static_cast<const AABBQuery*>(m_neighbor_query)
                          ->queryBallUnbounded(&(m_points[cur_p]), 1, m_r_cur);
                while (!ball_it->end())
                {
                    NeighborPoint np = ball_it->next();
                    if (np == NeighborQueryIterator::ITERATOR_TERMINATOR)
                        continue;

                    if (!m_exclude_ii || cur_p != np.ref_id)
                    {
                        np.id = cur_p;
                        // If we've expanded our search radius beyond safe
                        // distance, use the map instead of the vector.
                        if (m_search_extended)
                        {
                            if (!m_all_distances.count(np.ref_id) || m_all_distances[np.ref_id] > np.distance)
                            {
                                m_all_distances[np.ref_id] = np.distance;
                            }
                        }
                        else
                        {
                            m_current_neighbors.emplace_back(np);
                        }
                    }
                }

                // Break if there are enough neighbors, or if we are querying beyond the limits of
                // the periodic box.
                m_r_cur *= m_scale;

                if (m_current_neighbors.size() >= m_k)
                {
                    std::sort(m_current_neighbors.begin(), m_current_neighbors.end());
                    break;
                }
                else if ((m_r_cur >= max_plane_distance) || (m_all_distances.size() >= m_k))
                {
                    // Once this condition is reached, either we found enough
                    // neighbors beyond the normal min_plane_distance
                    // condition or we conclude that there are not enough
                    // neighbors left in the system.
                    for (std::map<unsigned int, float>::const_iterator it(m_all_distances.begin());
                         it != m_all_distances.end(); it++)
                    {
                        m_current_neighbors.emplace_back(cur_p, it->first, it->second);
                    }
                    std::sort(m_current_neighbors.begin(), m_current_neighbors.end());
                    break;
                }
                else if (m_r_cur > min_plane_distance / 2)
                {
                    // If we have to go beyond the cutoff radius, we need to
                    // start tracking what particles are already in the set so
                    // that we can make sure that we find the closest image
                    // because we now run the risk of finding duplicates.
                    // We could make this marginally more efficient by checking
                    // whether we've exactly hit the limit, or if there's a
                    // rescaling that would let us try the exact limit once
                    // before going beyond the min plane distance.
                    m_search_extended = true;
                    for (std::vector<NeighborPoint>::const_iterator it(m_current_neighbors.begin());
                         it != m_current_neighbors.end(); it++)
                    {
                        m_all_distances[it->ref_id] = it->distance;
                    }
                }
            }
        }

        // Now we return all the points found for the current point
        while ((m_count < m_k) && (m_count < m_current_neighbors.size()))
        {
            m_count++;
            return m_current_neighbors[m_count - 1];
        }

        cur_p++;
        m_count = 0;
        m_current_neighbors.clear();
        m_all_distances.clear();
        m_r_cur = m_r;
        m_search_extended = false;
    }
    m_finished = true;
    return NeighborQueryIterator::ITERATOR_TERMINATOR;
}

std::shared_ptr<NeighborQueryIterator> AABBQueryIterator::query(unsigned int idx)
{
    return this->m_aabb_query->query(&m_points[idx], 1, m_k, m_r, m_scale);
}
}; }; // end namespace freud::locality
