#include <boost/python.hpp>
#include <boost/shared_array.hpp>

#include "LinkCell.h"
#include "num_util.h"
#include "trajectory.h"

#ifndef _PMFXY2D_H__
#define _PMFXY2D_H__

/*! \internal
    \file PMFXY2D.h
    \brief Routines for computing radial density functions
*/

namespace freud { namespace pmft {

//! Computes the PCF for a given set of points
/*! A given set of reference points is given around which the PCF is computed and averaged in a sea of data points.
    Computing the PCF results in a pcf array listing the value of the PCF at each given x, y, listed in the x, y arrays.

    The values of x, y to compute the pcf at are controlled by the xmax, ymax and dx, dy parameters to the constructor.
    xmax, ymax determines the minimum/maximum x,y at which to compute the pcf and dx, dy is the step size for each bin.

    <b>2D:</b><br>
    This PCF only works for 2D boxes. As with everything else in freud, 2D points must be passed in as
    3 component vectors x,y,0. Failing to set 0 in the third component will lead to undefined behavior.
*/
class PMFXY2D
    {
    public:
        //! Constructor
        PMFXY2D(const trajectory::Box& box, float max_x, float max_y, float dx, float dy);

        //! Destructor
        ~PMFXY2D();

        //! Get the simulation box
        const trajectory::Box& getBox() const
            {
            return m_box;
            }

        //! Check if a cell list should be used or not
        bool useCells();

        void resetPCF();

        void resetPCFPy()
            {
            resetPCF();
            }

        /*! Compute the PCF for the passed in set of points. The function will be added to previous values
            of the pcf
        */
        void compute(float3 *ref_points,
                     float *ref_orientations,
                     unsigned int Nref,
                     float3 *points,
                     float *orientations,
                     unsigned int Np);

        //! Python wrapper for compute
        void computePy(boost::python::numeric::array ref_points,
                       boost::python::numeric::array ref_orientations,
                       boost::python::numeric::array points,
                       boost::python::numeric::array orientations);

        //! Get a reference to the PCF array
        boost::shared_array<unsigned int> getPCF()
            {
            return m_pcf_array;
            }

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
        boost::python::numeric::array getPCFPy()
            {
            unsigned int *arr = m_pcf_array.get();
            return num_util::makeNum(arr, m_nbins_x * m_nbins_y);
            }

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

        boost::shared_array<unsigned int> m_pcf_array;         //!< pcf array computed
        boost::shared_array<float> m_x_array;           //!< array of x values that the pcf is computed at
        boost::shared_array<float> m_y_array;           //!< array of y values that the pcf is computed at
    };

/*! \internal
    \brief Exports all classes in this file to python
*/
void export_PMFXY2D();

}; }; // end namespace freud::pmft

#endif // _PMFXY2D_H__
