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

#ifndef _PMFXYZ_H__
#define _PMFXYZ_H__

/*! \internal
    \file PMFXYZ.h
    \brief Routines for computing anisotropic potential of mean force in 3D
*/

namespace freud { namespace pmft {

//! Computes the PCF for a given set of points
/*! A given set of reference points is given around which the PCF is computed and averaged in a sea of data points.
    Computing the PCF results in a pcf array listing the value of the PCF at each given x, y, z listed in the x, y, and z arrays.

    The values of x, y, z to compute the pcf at are controlled by the xmax, ymax, zmax and dx, dy, dz parameters to the constructor.
    xmax, ymax, zmax determines the minimum/maximum x, y, z at which to compute the pcf and dx, dy, dz is the step size for each bin.

    <b>2D:</b><br>
    This PCF works for 3D boxes (while it will work for 2D boxes, you should use the 2D version).
*/
class PMFXYZ
    {
    public:
        //! Constructor
        PMFXYZ(float max_x, float max_y, float max_z, float dx, float dy, float dz);

        //! Destructor
        ~PMFXYZ();

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
        void compute(const vec3<float> *ref_points,
                     const quat<float> *ref_orientations,
                     unsigned int Nref,
                     const vec3<float> *points,
                     const quat<float> *orientations,
                     unsigned int Np,
                     const quat<float> *face_orientations,
                     const unsigned int Nfaces);

        //! Python wrapper for compute
        void computePy(trajectory::Box& box,
                       boost::python::numeric::array ref_points,
                       boost::python::numeric::array ref_orientations,
                       boost::python::numeric::array points,
                       boost::python::numeric::array orientations,
                       boost::python::numeric::array face_orientations);

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

        //! Get a reference to the z array
        boost::shared_array<float> getZ()
            {
            return m_z_array;
            }

        //! Python wrapper for getPCF() (returns a copy)
        boost::python::numeric::array getPCFPy()
            {
            unsigned int *arr = m_pcf_array.get();
            std::vector<intp> dims(3);
            dims[0] = m_nbins_z;
            dims[1] = m_nbins_y;
            dims[2] = m_nbins_x;
            return num_util::makeNum(arr, dims);
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

        //! Python wrapper for getZ() (returns a copy)
        boost::python::numeric::array getZPy()
            {
            float *arr = m_z_array.get();
            return num_util::makeNum(arr, m_nbins_z);
            }
    private:
        trajectory::Box m_box;            //!< Simulation box the particles belong in
        float m_max_x;                     //!< Maximum x at which to compute pcf
        float m_max_y;                     //!< Maximum y at which to compute pcf
        float m_max_z;                     //!< Maximum z at which to compute pcf
        float m_dx;                       //!< Step size for x in the computation
        float m_dy;                       //!< Step size for y in the computation
        float m_dz;                       //!< Step size for z in the computation
        locality::LinkCell* m_lc;          //!< LinkCell to bin particles for the computation
        unsigned int m_nbins_x;             //!< Number of x bins to compute pcf over
        unsigned int m_nbins_y;             //!< Number of y bins to compute pcf over
        unsigned int m_nbins_z;             //!< Number of z bins to compute pcf over

        boost::shared_array<unsigned int> m_pcf_array;         //!< array of pcf computed
        boost::shared_array<float> m_x_array;           //!< array of x values that the pcf is computed at
        boost::shared_array<float> m_y_array;           //!< array of y values that the pcf is computed at
        boost::shared_array<float> m_z_array;           //!< array of z values that the pcf is computed at
        tbb::enumerable_thread_specific<unsigned int *> m_local_pcf_array;
    };

/*! \internal
    \brief Exports all classes in this file to python
*/
void export_PMFXYZ();

}; }; // end namespace freud::pmft

#endif // _PMFXYZ_H__
