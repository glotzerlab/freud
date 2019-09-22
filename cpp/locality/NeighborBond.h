#ifndef NEIGHBOR_BOND_H
#define NEIGHBOR_BOND_H

namespace freud { namespace locality {

//! Simple data structure encoding neighboring points.
/*! The primary purpose of this class is to provide a more meaningful struct
 *  than a simple std::pair, which is hard to interpret. Additionally, this
 *  class defines the less than operator according to distance, making it
 *  possible to sort.
 */
struct NeighborBond
{
    // For now, id = query_point_idx and ref_id = point_idx (into the NeighborQuery).
    NeighborBond() : id(0), ref_id(0), distance(0),  weight(0) {}

    NeighborBond(unsigned int id, unsigned int ref_id, float d, float w) :
        id(id), ref_id(ref_id), distance(d), weight(w) {}

    NeighborBond(unsigned int id, unsigned int ref_id, float d) :
        id(id), ref_id(ref_id), distance(d), weight(1) {}

    //! Equality checks both id and distance.
    bool operator==(const NeighborBond& other)
    {
        return (id == other.id) && (ref_id == other.ref_id) && (distance == other.distance);
    }

    //! Not equals checks inverse of equality.
	bool operator!=(const NeighborBond& other) {
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
        if (id != n.id)
        {
            return id < n.id;
        }
        if (ref_id != n.ref_id)
        {
            return ref_id < n.ref_id;
        }
        return weight < n.weight;
    }

    bool less_as_tuple(const NeighborBond& n) const
    {
        if (id != n.id)
        {
            return id < n.id;
        }
        if (ref_id != n.ref_id)
        {
            return ref_id < n.ref_id;
        }
        if (weight != n.weight)
        {
            return weight < n.weight;
        }
        return distance < n.distance;
    }

    unsigned int id;          //! The point id.
    unsigned int ref_id;      //! The reference point id.
    float distance;           //! The distance between the points.
    float weight;             //! The weight of this bond.
};


}; }; // end namespace freud::locality

#endif // NEIGHBOR_BOND_H
