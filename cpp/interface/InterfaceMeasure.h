// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef INTERFACE_MEASURE_H
#define INTERFACE_MEASURE_H

#include <memory>

#include "box.h"
#include "VectorMath.h"
#include "LinkCell.h"

/*! \file InterfaceMeasure.h
    \brief Compute the size of an interface between two point clouds.
*/

namespace freud { namespace interface {

//! Computes the amount of interface for two given sets of points
/*! Given two sets of points, calculates the amount of points in the first
    set (reference) that are within a cutoff distance from any point in the
    second set.

    <b>2D:</b><br>
    InterfaceMeasure properly handles 2D boxes. As with everything else in
    freud, 2D points must be passed in as 3 component vectors x,y,0. Failing
    to set 0 in the third component will lead to undefined behavior.
 */
class InterfaceMeasure
{
    public:
        //! Constructor
        InterfaceMeasure(const box::Box& box, float r_cut);

        //! Get the simulation box
        const box::Box& getBox() const
        {
            return m_box;
        }

        //! Compute the interface
        void compute(const freud::locality::NeighborList *nlist,
                     const vec3<float> *ref_points,
                     unsigned int n_ref,
                     const vec3<float> *points,
                     unsigned int Np);

        //! Return the number of particles at the interface
        unsigned int getInterfaceCount()
            {
            return m_interface_count;
            }

        //! Get a reference to the last computed interface particle IDs
        std::shared_ptr< std::vector<unsigned int> > getInterfaceIds()
            {
            return m_interface_ids;
            }

    private:
        box::Box m_box;  //!< Simulation box where the particles belong
        float m_rcut;    //!< Max distance at which particles are considered to be in an interface
        unsigned int m_interface_count; //!< Number of particles at the interface
        std::shared_ptr< std::vector<unsigned int> > m_interface_ids; //!< IDs of particles at the interface
};

}; }; // end namespace freud::interface

#endif // INTERFACE_MEASURE_H
