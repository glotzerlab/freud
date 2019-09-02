// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef AABBQUERY_H
#define AABBQUERY_H

#include <map>
#include <memory>
#include <vector>

#include "AABBTree.h"
#include "Box.h"
#include "NeighborQuery.h"

/*! \file AABBQuery.h
    \brief Build an AABB tree from points and query it for neighbors.
 * A bounding volume hierarchy (BVH) tree is a binary search tree. It is
 * constructed from axis-aligned bounding boxes (AABBs). The AABB for a node in
 * the tree encloses all child AABBs. A leaf AABB holds multiple particles. The
 * tree is constructed in a balanced way using a heuristic to minimize AABB
 * volume. We build one tree per particle type, and use point AABBs for the
 * particles. The neighbor list is built by traversing down the tree with an
 * AABB that encloses the pairwise cutoff for the particle. Periodic boundaries
 * are treated by translating the query AABB by all possible image vectors,
 * many of which are trivially rejected for not intersecting the root node.
 */

namespace freud { namespace locality {

class AABBQuery : public NeighborQuery
{
public:
    //! Constructs the compute
    AABBQuery();

    //! New-style constructor.
    AABBQuery(const box::Box& box, const vec3<float>* points, unsigned int n_points);

    //! Destructor
    ~AABBQuery();

    //! Implementation of per-particle query for AABBQuery (see NeighborQuery.h for documentation).
    /*! \param query_point The point to find neighbors for.
     *  \param n_query_points The number of query points.
     *  \param qargs The query arguments that should be used to find neighbors.
     */
    virtual std::shared_ptr<NeighborQueryPerPointIterator> querySingle(const vec3<float> query_point, unsigned int query_point_idx,
                                                                 QueryArgs args) const;

    AABBTree m_aabb_tree; //!< AABB tree of points

protected:
    //! Validate the combination of specified arguments.
    /*! Add to parent function to account for the various arguments
     *  specifically required for AABBQuery nearest neighbor queries.
     */
    virtual void validateQueryArgs(QueryArgs& args) const
    {
        NeighborQuery::validateQueryArgs(args);
        if (args.mode == QueryArgs::nearest)
        {
            if (args.scale == QueryArgs::DEFAULT_SCALE)
            {
                args.scale = float(1.1);
            }
            if (args.r_max == QueryArgs::DEFAULT_R_MAX)
            {
                // By default, we use 1/10 the smallest box dimension as the guessed query distance.
                vec3<float> L = this->getBox().getL();
                float r_max = std::min(L.x, L.y);
                r_max = this->getBox().is2D() ? r_max : std::min(r_max, L.z);
                args.r_max = float(0.1) * r_max;
            }
        }
    }

private:
    //! Driver for tree configuration
    void setupTree(unsigned int N);

    //! Maps particles by local id to their id within their type trees
    void mapParticlesByType();

    //! Driver to build AABB trees
    void buildTree(const vec3<float>* points, unsigned int N);

    std::vector<AABB> m_aabbs; //!< Flat array of AABBs of all types
    box::Box m_box;            //!< Simulation box where the particles belong
};

//! Parent class of AABB iterators that knows how to traverse general AABB tree structures.
class AABBIterator : public NeighborQueryPerPointIterator
{
public:
    //! Constructor
    AABBIterator(const AABBQuery* neighbor_query, const vec3<float> query_point, unsigned int query_point_idx, bool exclude_ii)
        : NeighborQueryPerPointIterator(neighbor_query, query_point, query_point_idx, exclude_ii), m_aabb_query(neighbor_query)
    {}

    //! Empty Destructor
    virtual ~AABBIterator() {}

    //! Computes the image vectors to query for
    void updateImageVectors(float r_max, bool _check_r_max = true);

protected:
    const AABBQuery* m_aabb_query;         //!< Link to the AABBQuery object
    std::vector<vec3<float>> m_image_list; //!< List of translation vectors
    unsigned int m_n_images;               //!< The number of image vectors to check
};

//! Iterator that gets a specified number of nearest neighbors from AABB tree structures.
class AABBQueryIterator : public AABBIterator
{
public:
    //! Constructor
    AABBQueryIterator(const AABBQuery* neighbor_query, const vec3<float> query_point, unsigned int query_point_idx,
                      unsigned int num_neighbors, float r, float scale, bool exclude_ii)
        : AABBIterator(neighbor_query, query_point, query_point_idx, exclude_ii), m_count(0), m_num_neighbors(num_neighbors), m_search_extended(false), m_r_cur(r),
          m_scale(scale), m_all_distances()
    {
        updateImageVectors(0);
    }

    //! Empty Destructor
    virtual ~AABBQueryIterator() {}

    //! Get the next element.
    virtual NeighborBond next();

protected:
    unsigned int m_count;                           //!< Number of neighbors returned for the current point.
    unsigned int m_num_neighbors;                               //!< Number of nearest neighbors to find
    std::vector<NeighborBond> m_current_neighbors; //!< The current set of found neighbors.
    float m_search_extended; //!< Flag to see whether we've gone past the safe cutoff distance and have to be
                             //!< worried about finding duplicates.
    float
        m_r_cur; //!< Current search ball cutoff distance in use for the current particle (expands as needed).
    float m_scale; //!< The amount to scale m_r by when the current ball is too small.
    std::map<unsigned int, float> m_all_distances; //!< Hash map of minimum distances found for a given point,
                                                   //!< used when searching beyond maximum safe AABB distance.
};

//! Iterator that gets neighbors in a ball of size r_max using AABB tree structures.
class AABBQueryBallIterator : public AABBIterator
{
public:
    //! Constructor
    AABBQueryBallIterator(const AABBQuery* neighbor_query, const vec3<float> query_point, unsigned int query_point_idx, float r_max,
                          bool exclude_ii, bool _check_r_max = true)
        : AABBIterator(neighbor_query, query_point, query_point_idx, exclude_ii), m_r_max(r_max), cur_image(0), cur_node_idx(0),
          cur_ref_p(0)
    {
        updateImageVectors(m_r_max, _check_r_max);
    }

    //! Empty Destructor
    virtual ~AABBQueryBallIterator() {}

    //! Get the next element.
    virtual NeighborBond next();

protected:
    float m_r_max; //!< Search ball cutoff distance.

private:
    unsigned int cur_image;    //!< The current node in the tree.
    unsigned int cur_node_idx; //!< The current node in the tree.
    unsigned int
        cur_ref_p; //!< The current index into the reference particles in the current node of the tree.
};
}; }; // end namespace freud::locality

#endif // AABBQUERY_H
