#include <tbb/tbb.h>
#include <ostream>

// work around nasty issue where python #defines isalpha, toupper, etc....
#undef __APPLE__
#include <python.h>
#define __APPLE__

#include <boost/python.hpp>
#include <boost/shared_array.hpp>

#include "HOOMDMath.h"
#include "VectorMath.h"

#include "LinkCell.h"
#include "num_util.h"
#include "trajectory.h"
#include "Index1D.h"

#ifndef _lindemann_H__
#define _lindemann_H__

/*! \file lind.h
    \brief Routines for computing radial density functions
*/

namespace freud { namespace lindemann {

//! Computes the RDF (g(r)) for a given set of points
/*! A given set of reference points is given around which the RDF is computed and averaged in a sea of data points.
    Computing the RDF results in an rdf array listing the value of the RDF at each given r, listed in the r array.

    The values of r to compute the rdf at are controlled by the rmax and dr parameters to the constructor. rmax
    determins the maximum r at which to compute g(r) and dr is the step size for each bin.

    <b>2D:</b><br>
    RDF properly handles 2D boxes. As with everything else in freud, 2D points must be passed in as
    3 component vectors x,y,0. Failing to set 0 in the third component will lead to undefined behavior.
*/
class Lind
    {
    public:
        //! Constructor
        Lind(const trajectory::Box& box, float rmax, float dr);

        //! Get the simulation box
        const trajectory::Box& getBox() const
            {
            return m_box;
            }

        //! Compute the Lindemann Index
        // void compute(const float3 *points,
        //          unsigned int Np,
        //          unsigned int Nf);
        void compute(const vec3<float> *points,
                 unsigned int Np,
                 unsigned int Nf);

        //! Python wrapper for compute
        void computePy(boost::python::numeric::array points);

        //! Get a reference to the last computed rdf
        boost::shared_array< float > getLindexArray()
            {
            return m_lindex_array;
            }

        //! Python wrapper for getLindex() (returns a copy)
        boost::python::numeric::array getLindexArrayPy()
            {
            float *arr = m_lindex_array.get();
            return num_util::makeNum(arr, m_Np);
            }

    private:
        trajectory::Box m_box;            //!< Simulation box the particles belong in
        float m_rmax;                     //!< Maximum r at which to compute g(r)
        float m_dr;                       //!< Step size for r in the computation
        unsigned int m_Np;                //!< used only to return the array
        boost::shared_array< float > m_lindex_array;         //!< lindex array computed
    };

/*! \internal
    \brief Exports all classes in this file to python
*/
void export_lindemann();

}; }; // end namespace freud::lindemann

#endif // _lindemann_H__
