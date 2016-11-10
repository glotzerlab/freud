// Copyright (c) 2010-2016 The Regents of the University of Michigan
// This file is part of the Freud project, released under the BSD 3-Clause License.

#include <memory>

#include "HOOMDMath.h"
#include "VectorMath.h"

#include "box.h"
#include "LinkCell.h"

#ifndef _INTERFACEMEASURE_H_
#define _INTERFACEMEASURE_H_

/*! \file InterfaceMeasure.h
    \brief Compute the size of an interface between two point clouds
*/

namespace freud { namespace interface {

//! Computes the amount of interface for two given sets of points
/*! Given two sets of points, calculates the amount of points in the first set (reference) that are within a
 *  cutoff distance from any point in the second set.
 *
 *  <b>2D:</b><br>
 *  InterfaceMeasure properly handles 2D boxes. As with everything else in freud, 2D points must be passed in
 *  as 3 component vectors x,y,0. Failing to set 0 in the third component will lead to undefined behavior.
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
        // unsigned int compute(const float3 *ref_points,
        //                      unsigned int n_ref,
        //                      const float3 *points,
        //                      unsigned int Np);

        unsigned int compute(const vec3<float> *ref_points,
                             unsigned int n_ref,
                             const vec3<float> *points,
                             unsigned int Np);

        // //! Python wrapper for compute
        // unsigned int computePy(boost::python::numeric::array ref_points,
        //                      boost::python::numeric::array points);
    private:
        box::Box m_box;          //!< Simulation box the particles belong in
        float m_rcut;                   //!< Maximum distance at which a particle is considered to be in an interface
        locality::LinkCell m_lc;        //!< LinkCell to bin particles for the computation
};

}; }; // end namespace freud::interface

#endif // _INTERFACEMEASURE_H__
