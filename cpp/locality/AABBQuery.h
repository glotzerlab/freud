// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef AABBQUERY_H
#define AABBQUERY_H

#include <vector>

#include "SpatialData.h"
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

class AABBQuery : public SpatialData
    {
    public:
        //! Constructs the compute
        AABBQuery();

        //! New-style constructor.
        AABBQuery(const box::Box &box, const vec3<float> *ref_points, unsigned int Nref);

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

        //! Given a set of points, find the k elements of this data structure
        //  that are the nearest neighbors for each point.
        virtual SpatialDataIterator query(const vec3<float> *points, unsigned int Np, unsigned int k);

        //! Given a set of points, find all elements of this data structure
        //  that are within a certain distance r.
        virtual SpatialDataIterator query_ball(const vec3<float> *points, unsigned int Np, float r);

        AABBTree m_aabb_tree; //!< AABB tree of points

    private:
        //! Driver for tree configuration
        void setupTree(unsigned int N);

        //! Maps particles by local id to their id within their type trees
        void mapParticlesByType();

        //! Computes the image vectors to query for
        void updateImageVectors();

        //! Driver to build AABB trees
        void buildTree(const vec3<float> *ref_points, unsigned int N);

        //! Traverses AABB trees to compute neighbors
        void traverseTree(const vec3<float> *ref_points, unsigned int Nref,
            const vec3<float> *points, unsigned int Np, bool exclude_ii);

        unsigned int m_Ntotal;
        std::vector<AABB> m_aabbs; //!< Flat array of AABBs of all types
        std::vector< vec3<float> > m_image_list; //!< List of translation vectors
        unsigned int m_n_images; //!< The number of image vectors to check

        box::Box m_box; //!< Simulation box where the particles belong
        float m_rcut; //!< Maximum distance between neighbors
        NeighborList m_neighbor_list; //!< Stored neighbor list

        bool m_prebuilt; //! Flag for when constructed using the new style.
    };

//! Parent class of AABB iterators that knows how to traverse general AABB tree structures
/*! placeholder

*/
class AABBIterator : public SpatialDataIterator
    {
    public:
        //! Constructor
        AABBIterator(AABBQuery* spatial_data, const vec3<float> *points, unsigned int Np) : SpatialDataIterator(spatial_data, points, Np), m_aabb_data(spatial_data)
        { }

        //! Empty Destructor
        virtual ~AABBIterator() {}

        //! Computes the image vectors to query for
        void updateImageVectors(float rmax);

    protected:
        const AABBQuery *m_aabb_data; //!< Link to the AABBQuery object
        std::vector< vec3<float> > m_image_list; //!< List of translation vectors
        unsigned int m_n_images;                 //!< The number of image vectors to check
    };

//! Iterator that gets nearest neighbors from AABB tree structures
/*! placeholder

*/
class AABBQueryIterator : public AABBIterator
    {
    public:
        //! Constructor
        AABBQueryIterator(AABBQuery* spatial_data,
                const vec3<float> *points, unsigned int Np, unsigned int k) :
            AABBIterator(spatial_data, points, Np), m_k(k)
        {
        updateImageVectors(0);
        }

        //! Empty Destructor
        virtual ~AABBQueryIterator() {}

        //! Get the next element.
        //virtual std::pair<unsigned int, float> next();

    protected:
        unsigned int m_k;  //!< Number of nearest neighbors to find
    };

//! Iterator that gets neighbors in a ball of size r using AABB tree structures
/*! placeholder

*/
class AABBQueryBallIterator : public AABBIterator
    {
    public:
        //! Constructor
        AABBQueryBallIterator(AABBQuery* spatial_data,
                const vec3<float> *points, unsigned int Np, float r) :
            AABBIterator(spatial_data, points, Np), m_r(r), i(0), cur_image(0), cur_node_idx(0), cur_p(0)
        {
        updateImageVectors(m_r);
        }

        //! Empty Destructor
        virtual ~AABBQueryBallIterator() {}

        virtual bool end() { return m_done; }

        //! Get the next element.
        virtual std::pair<unsigned int, float> next();

    protected:
        float m_r;  //!< Search ball cutoff distance

    private:
        bool m_done;  //!< Done iterating.
        size_t i;   //!< The iterator over points
        unsigned int cur_image; 
        unsigned int cur_node_idx; 
        unsigned int cur_p; 


    };
}; }; // end namespace freud::locality

#endif // AABBQUERY_H
