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

// Forward declare the iterator
class SpatialDataIterator;


//! Parent data structure for all neighbor finding algorithms.
/*! placeholder

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
        virtual std::shared_ptr<SpatialDataIterator> query_ball(const vec3<float> point, float r) const = 0;

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

        static const std::pair<unsigned int, float> ITERATOR_TERMINATOR; //!< The object returned when iteration is complete.

    protected:
        const box::Box m_box;             //!< Simulation box where the particles belong
        const vec3<float> *m_ref_points;  //!< Reference point coordinates
        unsigned int m_Nref;              //!< Number of reference points

        
    };

//! The iterator class over the Spatial Data
/*! placeholder

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
        virtual std::pair<unsigned int, float> next()
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
