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

#ifndef _PMFTRPM_H__
#define _PMFTRPM_H__

/*! \file PMFTRPM.h
    \brief Routines for computing radial density functions
*/

namespace freud { namespace pmft {

//! Computes the RDF (g(r)) for a given set of points
/*! A given set of reference points is given around which the RDF is computed and averaged in a sea of data points.
    Computing the RDF results in an rdf array listing the value of the RDF at each given r, listed in the r array.

    The values of r to compute the rdf at are controlled by the rmax and dr parameters to the constructor. rmax
    determins the maximum r at which to compute g(r) and dr is the step size for each bin.

    <b>2D:</b><br>
    RDF properly handles 2D boxes. As with everything else in freud, 2D points must be passed in as
    3 component vectors x,y,0. Failing to set 0 in the third component will lead to undefined behavior.
*/
class PMFTRPM
    {
    public:
        //! Constructor
        PMFTRPM(float max_r, float max_TP, float max_TM, unsigned int nbins_r, unsigned int nbins_TP, unsigned int nbins_TM);

        //! Destructor
        ~PMFTRPM();

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

        //! Get a reference to the R array
        boost::shared_array<float> getR()
            {
            return m_r_array;
            }

        //! Get a reference to the TP array
        boost::shared_array<float> getTP()
            {
            return m_TP_array;
            }

        //! Get a reference to the TM array
        boost::shared_array<float> getTM()
            {
            return m_TM_array;
            }

        //! Python wrapper for getPCF() (returns a copy)
        boost::python::numeric::array getPCFPy();

        //! Python wrapper for getX() (returns a copy)
        boost::python::numeric::array getRPy()
            {
            float *arr = m_r_array.get();
            return num_util::makeNum(arr, m_nbins_r);
            }

        //! Python wrapper for getY() (returns a copy)
        boost::python::numeric::array getTPPy()
            {
            float *arr = m_TP_array.get();
            return num_util::makeNum(arr, m_nbins_TP);
            }

        //! Python wrapper for getT() (returns a copy)
        boost::python::numeric::array getTMPy()
            {
            float *arr = m_TM_array.get();
            return num_util::makeNum(arr, m_nbins_TM);
            }

    private:
        trajectory::Box m_box;            //!< Simulation box the particles belong in
        float m_max_r;                     //!< Maximum x at which to compute pcf
        float m_max_TP;                     //!< Maximum y at which to compute pcf
        float m_max_TM;                     //!< Maximum T at which to compute pcf
        float m_dr;                       //!< Step size for x in the computation
        float m_dTP;                       //!< Step size for y in the computation
        float m_dTM;                       //!< Step size for T in the computation
        locality::LinkCell* m_lc;          //!< LinkCell to bin particles for the computation
        unsigned int m_nbins_r;             //!< Number of x bins to compute pcf over
        unsigned int m_nbins_TP;             //!< Number of y bins to compute pcf over
        unsigned int m_nbins_TM;             //!< Number of T bins to compute pcf over

        boost::shared_array<unsigned int> m_pcf_array;         //!< array of pcf computed
        boost::shared_array<float> m_r_array;           //!< array of x values that the pcf is computed at
        boost::shared_array<float> m_TP_array;           //!< array of y values that the pcf is computed at
        boost::shared_array<float> m_TM_array;           //!< array of T values that the pcf is computed at
        tbb::enumerable_thread_specific<unsigned int *> m_local_pcf_array;
    };

/*! \internal
    \brief Exports all classes in this file to python
*/
void export_PMFTRPM();

}; }; // end namespace freud::pmft

#endif // _PMFTRPM_H__
