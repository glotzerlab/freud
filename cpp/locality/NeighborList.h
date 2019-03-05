// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef NEIGHBOR_LIST_H
#define NEIGHBOR_LIST_H

#include <memory>

#include "Box.h"
#include "VectorMath.h"

namespace freud { namespace locality {

//! Store a number of near-neighbor bonds from one set of positions (A, "reference points") to another set (B, "target points")
/*! A NeighborList object acts as a source of neighbor information for
    freud's compute methods. Briefly, each bond is associated with a
    reference point index, a target point index, and a weight.

    <b>Data structures:</b>

    Reference and target point indices are stored in an array
    (m_neighbors) as follows: for each bond index b, neighbors[2*b]
    yields the reference point index i and neighbors[2*b + 1] yields
    the target point index j. The weights array (m_weights) is a flat
    per-bond array of bond weights.

 */
class NeighborList
    {
    public:
        //! Default constructor
        NeighborList();
        //! Create a NeighborList that can hold up to the given number of bonds
        NeighborList(size_t max_bonds);
        //! Copy constructor
        NeighborList(const NeighborList &other);

        //! Return the number of bonds stored in this NeighborList
        size_t getNumBonds() const;
        //! Return the number of reference points this NeighborList was built with
        size_t getNumI() const;
        //! Return the number of target points this NeighborList was built with
        size_t getNumJ() const;
        //! Set the number of bonds, reference points, and target points for this NeighborList object
        void setNumBonds(size_t num_bonds, size_t num_i, size_t num_j);
        //! Access the neighbors array for reading and writing
        size_t *getNeighbors();
        //! Access the weights array for reading and writing
        float *getWeights();

        //! Access the neighbors array for reading
        const size_t *getNeighbors() const;
        //! Access the weights array for reading
        const float *getWeights() const;
        //! Remove bonds in this object based on an array of boolean values. The array must contain at least m_num_bonds elements.
        size_t filter(const bool *filt);
        //! Remove bonds in this object based on minimum and maximum distance constraints and the given position arrays r_i and r_j. r_i and r_j must have m_num_i and m_num_j elements, respectively.
        size_t filter_r(const freud::box::Box &box, const vec3<float> *r_i,
            const vec3<float> *r_j, float rmax, float rmin=0);

        //! Return the first bond index corresponding to reference point i
        size_t find_first_index(size_t i) const;

        //! Resize our member arrays to a larger size if given (always resize to exactly the given size if force is true)
        void resize(size_t max_bonds, bool force=false);
        //! Copy the bonds from another NeighborList object
        void copy(const NeighborList &other);
        //! Throw a runtime_error if num_i and num_j do not match our stored value
        void validate(size_t num_i, size_t num_j) const;
    private:
        //! Helper method for bisection search of the neighbor list, used in find_first_index
        size_t bisection_search(size_t val, size_t left, size_t right) const;

        //! Maximum number of bonds we have allocated storage for
        size_t m_max_bonds;
        //! Number of bonds we are currently holding
        size_t m_num_bonds;
        //! Number of reference points
        size_t m_num_i;
        //! Number of target points
        size_t m_num_j;
        //! Neighbor list indices array
        std::shared_ptr<size_t> m_neighbors;
        //! Neighbor list per-bond weight array
        std::shared_ptr<float> m_weights;
    };

}; }; // end namespace freud::locality

#endif // NEIGHBOR_LIST_H
