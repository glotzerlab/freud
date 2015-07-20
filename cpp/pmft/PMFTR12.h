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

#ifndef _PMFTR12_H__
#define _PMFTR12_H__

/*! \file PMFTR12.h
    \brief Routines for computing radial density functions
*/

namespace freud { namespace pmft {

//! Computes the PCF for a given set of points
/*! A given set of reference points is given around which the PCF is computed and averaged in a sea of data points.
    Computing the PCF results in a pcf array listing the value of the PCF at each given x, y, z listed in the x, y, and z arrays.

    The values of r, T1, T2 to compute the pcf at are controlled by the rmax, T1max, T2max and nbins_r, nbins_T1, nbins_T2 parameters to the constructor.
    rmax, T1max, T2max determines the minimum/maximum r, T1, T2 at which to compute the pcf and nbins_r, nbins_T1, nbins_T2 is the number of bins in r, T1, T2.

    <b>2D:</b><br>
    This PCF works for 3D boxes (while it will work for 2D boxes, you should use the 2D version).
*/
class PMFTR12
    {
    public:
        //! Constructor
        PMFTR12(float max_r, float max_T1, float max_T2, unsigned int nbins_r, unsigned int nbins_T1, unsigned int nbins_T2);

        //! Destructor
        ~PMFTR12();

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

        //! Python wrapper for compute
        void computePy(trajectory::Box& box,
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

        //! Get a reference to the T1 array
        boost::shared_array<float> getT1()
            {
            return m_T1_array;
            }

        //! Get a reference to the T2 array
        boost::shared_array<float> getT2()
            {
            return m_T2_array;
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
        boost::python::numeric::array getT1Py()
            {
            float *arr = m_T1_array.get();
            return num_util::makeNum(arr, m_nbins_T1);
            }

        //! Python wrapper for getT() (returns a copy)
        boost::python::numeric::array getT2Py()
            {
            float *arr = m_T2_array.get();
            return num_util::makeNum(arr, m_nbins_T2);
            }

    private:
        trajectory::Box m_box;            //!< Simulation box the particles belong in
        float m_max_r;                     //!< Maximum x at which to compute pcf
        float m_max_T1;                     //!< Maximum y at which to compute pcf
        float m_max_T2;                     //!< Maximum T at which to compute pcf
        float m_dr;                       //!< Step size for x in the computation
        float m_dT1;                       //!< Step size for y in the computation
        float m_dT2;                       //!< Step size for T in the computation
        locality::LinkCell* m_lc;          //!< LinkCell to bin particles for the computation
        unsigned int m_nbins_r;             //!< Number of x bins to compute pcf over
        unsigned int m_nbins_T1;             //!< Number of y bins to compute pcf over
        unsigned int m_nbins_T2;             //!< Number of T bins to compute pcf over

        boost::shared_array<unsigned int> m_pcf_array;         //!< array of pcf computed
        boost::shared_array<float> m_r_array;           //!< array of x values that the pcf is computed at
        boost::shared_array<float> m_T1_array;           //!< array of y values that the pcf is computed at
        boost::shared_array<float> m_T2_array;           //!< array of T values that the pcf is computed at
        tbb::enumerable_thread_specific<unsigned int *> m_local_pcf_array;
    };

/*! \internal
    \brief Exports all classes in this file to python
*/
void export_PMFTR12();

}; }; // end namespace freud::pmft

#endif // _PMFTR12_H__
