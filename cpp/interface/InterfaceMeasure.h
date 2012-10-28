#include <boost/python.hpp>
#include <boost/shared_array.hpp>

#include "trajectory.h"
#include "LinkCell.h"
#include "num_util.h"

#ifndef _INTERFACEMEASURE_H_
#define _INTERFACEMEASURE_H_

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
        InterfaceMeasure(const trajectory::Box& box, float r_cut);

        //! Get the simulation box
        const trajectory::Box& getBox() const
        {
            return m_box;
        }

        //! Compute the interface
        unsigned int compute(const float3 *ref_points,
                             unsigned int Nref,
                             const float3 *points,
                             unsigned int Np);

        //! Python wrapper for compute
        unsigned int computePy(boost::python::numeric::array ref_points,
                             boost::python::numeric::array points);
    private:
        trajectory::Box m_box;          //!< Simulation box the particles belong in
        float m_rcut;                   //!< Maximum distance at which a particle is considered to be in an interface
        locality::LinkCell m_lc;        //!< LinkCell to bin particles for the computation
};

/*! \internal
 *  \brief Exports all classes in this file to python
 */
void export_InterfaceMeasure();

}; }; // end namespace freud::interface

#endif // _INTERFACEMEASURE_H__


