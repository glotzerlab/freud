// Copyright (c) 2010-2026 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef NEIGHBOR_BOND_H
#define NEIGHBOR_BOND_H

#include "VectorMath.h"
#include <cmath>

namespace freud { namespace locality {

//! Simple data structure encoding neighboring points.
/*! The primary purpose of this class is to provide a more meaningful struct
 *  than a simple std::pair, which is hard to interpret. Additionally, this
 *  class defines the less than operator according to distance, making it
 *  possible to sort.
 */
class NeighborBond
{
public:
    // For now, id = query_point_idx and ref_id = point_idx (into the NeighborQuery).
    constexpr NeighborBond() = default;

    /*
     * This constructor should only be used in cases where the correct distance/vector
     * pair is known before instantiating this object to save time and eliminate
     * redundancy. Common cases include NeighbQuery objects which have to compute
     * the distance to know if a neighbor pair fits the query arguments and NeighborList
     * objects which have already computed the distances corresponding to their vectors.
     * */
    constexpr NeighborBond(unsigned int query_point_idx, unsigned int point_idx, float d, float w,
                           const vec3<float>& v)
        : query_point_idx(query_point_idx), point_idx(point_idx), distance(d), weight(w), vector(v)
    {}

    /**
     * This constructor should be used for the majority of cases when instantiating
     * a NeighborBond object.
     */
    NeighborBond(unsigned int query_point_idx, unsigned int point_idx, float w, const vec3<float>& v)
        : query_point_idx(query_point_idx), point_idx(point_idx), distance(std::sqrt(dot(v, v))), weight(w),
          vector(v)
    {}

    //! Equality checks both query_point_idx and distance.
    bool operator==(const NeighborBond& other) const
    {
        return (query_point_idx == other.query_point_idx) && (point_idx == other.point_idx)
            && (distance == other.distance) && (vector == other.vector);
    }

    //! Not equals checks inverse of equality.
    bool operator!=(const NeighborBond& other) const
    {
        return !(*this == other);
    }

    //! Default comparator of points is by distance.
    /*! This form of comparison allows easy sorting of nearest neighbors by
     *  distance.
     */
    bool operator<(const NeighborBond& n) const
    {
        return distance < n.distance;
    }

    bool less_id_ref_weight(const NeighborBond& n) const
    {
        if (query_point_idx != n.query_point_idx)
        {
            return query_point_idx < n.query_point_idx;
        }
        if (point_idx != n.point_idx)
        {
            return point_idx < n.point_idx;
        }
        return weight < n.weight;
    }

    bool less_as_tuple(const NeighborBond& n) const
    {
        if (query_point_idx != n.query_point_idx)
        {
            return query_point_idx < n.query_point_idx;
        }
        if (point_idx != n.point_idx)
        {
            return point_idx < n.point_idx;
        }
        if (weight != n.weight)
        {
            return weight < n.weight;
        }
        return distance < n.distance;
    }

    bool less_as_distance(const NeighborBond& n) const
    {
        if (query_point_idx != n.query_point_idx)
        {
            return query_point_idx < n.query_point_idx;
        }
        if (distance != n.distance)
        {
            return distance < n.distance;
        }
        if (point_idx != n.point_idx)
        {
            return point_idx < n.point_idx;
        }
        return weight < n.weight;
    }

    unsigned int getQueryPointIdx() const
    {
        return query_point_idx;
    }

    void setQueryPointIdx(unsigned int idx)
    {
        query_point_idx = idx;
    }

    unsigned int getPointIdx() const
    {
        return point_idx;
    }

    void setPointIdx(unsigned int new_pidx)
    {
        point_idx = new_pidx;
    }

    float getWeight() const
    {
        return weight;
    }

    void setWeight(float w)
    {
        weight = w;
    }

    const vec3<float>& getVector() const
    {
        return vector;
    }

    void setVector(const vec3<float>& v)
    {
        vector = v;
        distance = std::sqrt(dot(v, v));
    }

    float getDistance() const
    {
        return distance;
    }

private:
    unsigned int query_point_idx {0}; //! The query point index.
    unsigned int point_idx {0};       //! The reference point index.
    float distance {0};               //! The distance between the points.
    float weight {0};                 //! The weight of this bond.
    vec3<float> vector;               //! The directed vector from query point to reference point.
};

}; }; // end namespace freud::locality

#endif // NEIGHBOR_BOND_H
