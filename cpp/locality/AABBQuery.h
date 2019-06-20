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
    AABBQuery(const box::Box& box, const vec3<float>* ref_points, unsigned int Nref);

    //! Destructor
    ~AABBQuery();

    //! Perform a query based on a set of query parameters.
    /*! Given a QueryArgs object and a set of points to perform a query
     *  with, this function will dispatch the query to the appropriate
     *  querying function. We override the parent function to support
     *  calling the `query` method with the correct signature.
     *
     *  This function should just be called query, but Cython's function
     *  overloading abilities seem buggy at best, so it's easiest to just
     *  rename the function.
     */
    virtual std::shared_ptr<NeighborQueryIterator> queryWithArgs(const vec3<float>* points, unsigned int N,
                                                                 QueryArgs args)
    {
        this->validateQueryArgs(args);
        if (args.mode == QueryArgs::ball)
        {
            return queryBall(points, N, args.rmax, args.exclude_ii);
        }
        else if (args.mode == QueryArgs::nearest)
        {
            return query(points, N, args.nn, args.rmax, args.scale, args.exclude_ii);
        }
        else
        {
            throw std::runtime_error("Invalid query mode provided to generic query function.");
        }
    }

    //! Given a set of points, find the k elements of this data structure
    //  that are the nearest neighbors for each point. Note that due to the
    //  different signature, this is not directly overriding the original
    //  method in NeighborQuery, so we have to explicitly invalidate calling
    //  with that signature.
    virtual std::shared_ptr<NeighborQueryIterator> query(const vec3<float>* points, unsigned int N,
                                                         unsigned int k, bool exclude_ii = false) const
    {
        throw std::runtime_error("AABBQuery k-nearest-neighbor queries must use the function signature that "
                                 "provides rmax and scale guesses.");
    }

    std::shared_ptr<NeighborQueryIterator> query(const vec3<float>* points, unsigned int N, unsigned int k,
                                                 float r, float scale, bool exclude_ii = false) const;

    //! Given a set of points, find all elements of this data structure
    //  that are within a certain distance r.
    virtual std::shared_ptr<NeighborQueryIterator> queryBall(const vec3<float>* points, unsigned int N,
                                                             float r, bool exclude_ii = false) const;

    //! Given a set of points, find all elements of this data structure
    //  that are within a certain distance r, even if that distance is
    //  larger than the normally allowed distance for AABB tree-based
    //  queries. Such queries will experience performance losses, but they
    //  are necessary to support k-nearest neighbor queries. This function
    //  is declared separately rather than as a simple extra parameter to
    //  queryBall to avoid complexities with interfering with the virtual
    //  inherited API that is exported to Cython.
    std::shared_ptr<NeighborQueryIterator> queryBallUnbounded(const vec3<float>* points, unsigned int N,
                                                              float r, bool exclude_ii = false) const;

    AABBTree m_aabb_tree; //!< AABB tree of points

protected:
    virtual void validateQueryArgs(QueryArgs& args)
    {
        if (args.mode == QueryArgs::ball)
        {
            if (args.rmax == -1)
                throw std::runtime_error("You must set rmax in the query arguments.");
        }
        else if (args.mode == QueryArgs::nearest)
        {
            if (args.nn == -1)
                throw std::runtime_error("You must set nn in the query arguments.");
            if (args.scale == -1)
            {
                args.scale = 1.1;
            }
            if (args.rmax == -1)
            {
                vec3<float> L = this->getBox().getL();
                float rmax = std::min(L.x, L.y);
                args.rmax = this->getBox().is2D() ? 0.1 * rmax : 0.1 * std::min(rmax, L.z);
            }
        }
    }

private:
    //! Driver for tree configuration
    void setupTree(unsigned int N);

    //! Maps particles by local id to their id within their type trees
    void mapParticlesByType();

    //! Driver to build AABB trees
    void buildTree(const vec3<float>* ref_points, unsigned int N);

    std::vector<AABB> m_aabbs; //!< Flat array of AABBs of all types
    box::Box m_box;            //!< Simulation box where the particles belong
};

//! Parent class of AABB iterators that knows how to traverse general AABB tree structures
class AABBIterator : virtual public NeighborQueryIterator
{
public:
    //! Constructor
    AABBIterator(const AABBQuery* neighbor_query, const vec3<float>* points, unsigned int N, bool exclude_ii)
        : NeighborQueryIterator(neighbor_query, points, N, exclude_ii), m_aabb_query(neighbor_query)
    {}

    //! Empty Destructor
    virtual ~AABBIterator() {}

    //! Computes the image vectors to query for
    void updateImageVectors(float rmax, bool _check_rmax = true);

protected:
    const AABBQuery* m_aabb_query;         //!< Link to the AABBQuery object
    std::vector<vec3<float>> m_image_list; //!< List of translation vectors
    unsigned int m_n_images;               //!< The number of image vectors to check
};

//! Iterator that gets nearest neighbors from AABB tree structures
class AABBQueryIterator : virtual public NeighborQueryQueryIterator, virtual public AABBIterator
{
public:
    //! Constructor
    AABBQueryIterator(const AABBQuery* neighbor_query, const vec3<float>* points, unsigned int N,
                      unsigned int k, float r, float scale, bool exclude_ii)
        : NeighborQueryIterator(neighbor_query, points, N, exclude_ii),
          NeighborQueryQueryIterator(neighbor_query, points, N, exclude_ii, k),
          AABBIterator(neighbor_query, points, N, exclude_ii), m_search_extended(false), m_r(r), m_r_cur(r),
          m_scale(scale), m_all_distances()
    {
        updateImageVectors(0);
    }

    //! Empty Destructor
    virtual ~AABBQueryIterator() {}

    //! Get the next element.
    virtual NeighborPoint next();

    //! Create an equivalent new query iterator on a per-particle basis.
    virtual std::shared_ptr<NeighborQueryIterator> query(unsigned int idx);

protected:
    float m_search_extended; //!< Flag to see whether we've gone past the safe cutoff distance and have to be
                             //!< worried about finding duplicates.
    float m_r;               //!< Ball cutoff distance. Used as a guess.
    float
        m_r_cur; //!< Current search ball cutoff distance in use for the current particle (expands as needed).
    float m_scale; //!< The amount to scale m_r by when the current ball is too small.
    std::map<unsigned int, float> m_all_distances; //!< Hash map of minimum distances found for a given point,
                                                   //!< used when searching beyond maximum safe AABB distance.
};

//! Iterator that gets neighbors in a ball of size r using AABB tree structures
class AABBQueryBallIterator : virtual public AABBIterator
{
public:
    //! Constructor
    AABBQueryBallIterator(const AABBQuery* neighbor_query, const vec3<float>* points, unsigned int N, float r,
                          bool exclude_ii, bool _check_rmax = true)
        : NeighborQueryIterator(neighbor_query, points, N, exclude_ii),
          AABBIterator(neighbor_query, points, N, exclude_ii), m_r(r), cur_image(0), cur_node_idx(0),
          cur_ref_p(0)
    {
        updateImageVectors(m_r, _check_rmax);
    }

    //! Empty Destructor
    virtual ~AABBQueryBallIterator() {}

    //! Get the next element.
    virtual NeighborPoint next();

    //! Create an equivalent new query iterator on a per-particle basis.
    virtual std::shared_ptr<NeighborQueryIterator> query(unsigned int idx);

protected:
    float m_r; //!< Search ball cutoff distance.

private:
    unsigned int cur_image;    //!< The current node in the tree.
    unsigned int cur_node_idx; //!< The current node in the tree.
    unsigned int
        cur_ref_p; //!< The current index into the reference particles in the current node of the tree.
};
}; }; // end namespace freud::locality

#endif // AABBQUERY_H
