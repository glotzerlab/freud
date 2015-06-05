#include <tbb/tbb.h>
#include <ostream>

// work around nasty issue where python #defines isalpha, toupper, etc....
#undef __APPLE__
#include <Python.h>
#define __APPLE__

#include <boost/python.hpp>
#include <boost/shared_array.hpp>

#include "HOOMDMath.h"
#include "VectorMath.h"

#include "LinkCell.h"
#include "num_util.h"
#include "trajectory.h"
#include "Index1D.h"

#ifndef _PMFXY2D_H__
#define _PMFXY2D_H__

/*! \internal
    \file PMFXY2D.h
    \brief Routines for computing anisotropic potential of mean force in 2D
*/

namespace freud { namespace pmft {

//! Computes the PCF for a given set of points
/*! A given set of reference points is given around which the PCF is computed and averaged in a sea of data points.
    Computing the PCF results in a pcf array listing the value of the PCF at each given x, y, listed in the x, y arrays.

    The values of x, y to compute the pcf at are controlled by the xmax, ymax and dx, dy parameters to the constructor.
    xmax, ymax determines the minimum/maximum x, y at which to compute the pcf and dx, dy is the step size for each bin.

    <b>2D:</b><br>
    This PCF only works for 2D boxes. As with everything else in freud, 2D points must be passed in as
    3 component vectors x,y,0. Failing to set 0 in the third component should not matter as the code forces z=0.
    However, this could still lead to undefined behavior and should be avoided anyway.
*/
class PMFXY2D
    {
    public:
        //! Constructor
        PMFXY2D(float max_x, float max_y, unsigned int nbins_x, unsigned int nbins_y);

        //! Destructor
        ~PMFXY2D();

        //! Get the simulation box
        const trajectory::Box& getBox() const
            {
            return m_box;
            }

        //! Reset the PCF array to all zeros
        void resetPCF();

        //! Python wrapper for reset method
        void resetPCFPy()
            {
            resetPCF();
            }

        /*! Compute the PCF for the passed in set of points. The function will be added to previous values
            of the pcf
        */
        void accumulate(vec3<float> *ref_points,
                        float *ref_orientations,
                        unsigned int Nref,
                        vec3<float> *points,
                        float *orientations,
                        unsigned int Np);

        //! Python wrapper for accumulate
        void accumulatePy(trajectory::Box& box,
                          boost::python::numeric::array ref_points,
                          boost::python::numeric::array ref_orientations,
                          boost::python::numeric::array points,
                          boost::python::numeric::array orientations);

        //! \internal
        //! helper function to reduce the thread specific arrays into the boost array
        void reducePCF();

        //! Get a reference to the PCF array
        boost::shared_array<unsigned int> getPCF();

        //! Get a reference to the x array
        boost::shared_array<float> getX()
            {
            return m_x_array;
            }

        //! Get a reference to the y array
        boost::shared_array<float> getY()
            {
            return m_y_array;
            }

        //! Python wrapper for getPCF() (returns a copy)
        boost::python::numeric::array getPCFPy();

        //! Python wrapper for getX() (returns a copy)
        boost::python::numeric::array getXPy()
            {
            float *arr = m_x_array.get();
            return num_util::makeNum(arr, m_nbins_x);
            }

        //! Python wrapper for getY() (returns a copy)
        boost::python::numeric::array getYPy()
            {
            float *arr = m_y_array.get();
            return num_util::makeNum(arr, m_nbins_y);
            }

    private:
        trajectory::Box m_box;            //!< Simulation box the particles belong in
        float m_max_x;                     //!< Maximum x at which to compute pcf
        float m_max_y;                     //!< Maximum y at which to compute pcf
        float m_dx;                       //!< Step size for x in the computation
        float m_dy;                       //!< Step size for y in the computation
        locality::LinkCell* m_lc;          //!< LinkCell to bin particles for the computation
        unsigned int m_nbins_x;             //!< Number of x bins to compute pcf over
        unsigned int m_nbins_y;             //!< Number of y bins to compute pcf over

        boost::shared_array<unsigned int> m_pcf_array;         //!< array of pcf computed
        boost::shared_array<float> m_x_array;           //!< array of x values that the pcf is computed at
        boost::shared_array<float> m_y_array;           //!< array of y values that the pcf is computed at
        tbb::enumerable_thread_specific<unsigned int *> m_local_pcf_array;
    };

/*! \internal
    \brief Exports all classes in this file to python
*/
void export_PMFXY2D();

}; }; // end namespace freud::pmft

#endif // _PMFXY2D_H__
