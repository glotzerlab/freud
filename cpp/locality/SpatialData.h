// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef SPATIAL_DATA_H
#define SPATIAL_DATA_H

#include "Box.h"
#include <stdexcept>
#include <memory>

/*! \file SpatialData.h
    \brief Defines the abstract API for collections of points that can be
           queried against for neighbors.
*/

namespace freud { namespace locality {


//! Simple data structure encoding neighboring points.
/*! The primary purpose of this class is to provide a more meaningful struct
 *  than a simple std::pair, which is hard to interpret. Additionally, this
 *  class defines the less than operator according to distance, making it
 *  possible to sort.
 */
struct NeighborPoint {
    NeighborPoint() : id(0), distance(0) { }

    NeighborPoint(unsigned int id, float d) : id(id), distance(d) { }

    //! Equality checks both id and distance.
    bool operator== (const NeighborPoint &n)
        {
        return (id == n.id) && (distance == n.distance);
        }

    //! Default comparator of points is by distance.
    /*! This form of comparison allows easy sorting of nearest neighbors by
     *  distance
     */
    bool operator< (const NeighborPoint &n) const
        {
        return distance < n.distance;
        }

    unsigned int id;
    float distance;
};


// Forward declare the iterator
class SpatialDataIterator;


//! Parent data structure for all neighbor finding algorithms.
/*! This class defines the API for all data structures for accelerating
 *  neighbor finding. The object encapsulates a set of points and a system box
 *  that define the set of points to search and the periodic system within these
 *  points can be found.
 *
 *  The primary interface to this class is through the query and queryBall
 *  methods, which support k-nearest neighbor queries and distance-based
 *  queries, respectively.
 */
class SpatialData
    {
    public:
        //! Nullary constructor for Cython
        SpatialData()
            {
            }

        //! Constructor
        SpatialData(const box::Box &box, const vec3<float> *ref_points, unsigned int Nref) :
            m_box(box), m_ref_points(ref_points), m_Nref(Nref)
            {
            }

        //! Empty Destructor
        virtual ~SpatialData() {}

        //! Given a point, find the k elements of this data structure
        //  that are the nearest neighbors for each point.
        virtual std::shared_ptr<SpatialDataIterator> query(const vec3<float> point, unsigned int k) const = 0;

        //! Given a point, find all elements of this data structure
        //  that are within a certain distance r.
        virtual std::shared_ptr<SpatialDataIterator> queryBall(const vec3<float> point, float r) const = 0;

        //! Get the simulation box
        const box::Box& getBox() const
            {
            return m_box;
            }

        //! Get the reference points
        const vec3<float>* getRefPoints() const
            {
            return m_ref_points;
            }

        //! Get the number of reference points
        const unsigned int getNRef() const
            {
            return m_Nref;
            }

        //! Get a point's coordinates using index operator notation
        const vec3<float> operator [](unsigned int index) const
            {
            if (index >= m_Nref)
                {
                throw std::runtime_error("SpatialData attempted to access a point with index >= Nref.");
                }
            return m_ref_points[index];
            }

        static const NeighborPoint ITERATOR_TERMINATOR; //!< The object returned when iteration is complete.

    protected:
        const box::Box m_box;             //!< Simulation box where the particles belong
        const vec3<float> *m_ref_points;  //!< Reference point coordinates
        unsigned int m_Nref;              //!< Number of reference points
    };

//! The iterator class for neighbor queries on SpatialData objects.
/*! This is an abstract class that defines the abstract API for neighbor
 *  iteration. All subclasses of SpatialData should also subclass
 *  SpatialDataIterator and define the next() method appropriately. The next()
 *  method is the primary mode of interaction with the iterator, and allows
 *  looping through the iterator.
 *
 *  Note that due to the fact that there is no way to know when iteration is
 *  complete until all relevant points are actually checked (irrespective of the
 *  underlying data structure), the end() method will not return true until the
 *  next method reaches the end of control flow at least once without finding a
 *  next neighbor. As a result, the next() method is required to return
 *  SpatialData::ITERATOR_TERMINATOR on all calls after the last neighbor is
 *  found in order to guarantee that the correct set of neighbors is considered.
*/
class SpatialDataIterator {
    public:
        //! Nullary constructor for Cython
        SpatialDataIterator()
            {
            }

        //! Constructor
        SpatialDataIterator(const SpatialData* spatial_data,
                const vec3<float> point) :
            m_spatial_data(spatial_data), m_point(point), m_finished(false)
            {
            }

        //! Empty Destructor
        virtual ~SpatialDataIterator() {}

        //! Indicate when done.
        virtual bool end() { return m_finished; }

        //! Get the next element.
        virtual NeighborPoint next()
            {
            throw std::runtime_error("The next method must be implemented by child classes.");
            }

    protected:
        const SpatialData *m_spatial_data; //!< Link to the SpatialData object
        const vec3<float> m_point;        //!< Query point coordinates

        unsigned int m_finished;           //!< Flag to indicate that iteration is complete (must be set by next on termination).
};

}; }; // end namespace freud::locality

#endif // SPATIAL_DATA_H
