// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef AABBQUERY_H
#define AABBQUERY_H

#include <vector>

#include "Box.h"
#include "NeighborList.h"
#include "Index1D.h"
#include "AABBTree.h"

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

class AABBQuery
    {
    public:
        //! Constructs the compute
        AABBQuery();

        //! Destructor
        ~AABBQuery();

        void compute(box::Box& box, float rcut,
            const vec3<float> *ref_points, unsigned int Nref,
            const vec3<float> *points, unsigned int Np,
            bool exclude_ii);

        freud::locality::NeighborList *getNeighborList()
            {
            return &m_neighbor_list;
            }

    private:
        //! Driver for tree configuration
        void setupTree(unsigned int Np);

        //! Maps particles by local id to their id within their type trees
        void mapParticlesByType();

        //! Computes the image vectors to query for
        void updateImageVectors();

        //! Driver to build AABB trees
        void buildTree(const vec3<float> *points, unsigned int Np);

        //! Traverses AABB trees to compute neighbors
        void traverseTree(const vec3<float> *ref_points, unsigned int Nref,
            const vec3<float> *points, unsigned int Np, bool exclude_ii);

        unsigned int m_Ntotal;
        AABBTree m_aabb_tree; //!< AABB tree of points
        std::vector<AABB> m_aabbs; //!< Flat array of AABBs of all types
        std::vector< vec3<float> > m_image_list; //!< List of translation vectors
        unsigned int m_n_images; //!< The number of image vectors to check

        box::Box m_box; //!< Simulation box where the particles belong
        float m_rcut; //!< Maximum distance between neighbors
        NeighborList m_neighbor_list; //!< Stored neighbor list
    };

}; }; // end namespace freud::locality

#endif // AABBQUERY_H
