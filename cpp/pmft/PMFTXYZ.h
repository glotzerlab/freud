// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is part of the freud project, released under the BSD 3-Clause License.

#include <tbb/tbb.h>
#include <ostream>

// work around nasty issue where python #defines isalpha, toupper, etc....
#undef __APPLE__
#include <Python.h>
#define __APPLE__

#include <memory>

#include "HOOMDMath.h"
#include "VectorMath.h"

#include "LinkCell.h"
#include "box.h"
#include "Index1D.h"

#ifndef _PMFTXYZ_H__
#define _PMFTXYZ_H__

/*! \internal
    \file PMFTXYZ.h
    \brief Routines for computing anisotropic potential of mean force in 3D
*/

namespace freud { namespace pmft {

//! Computes the PCF for a given set of points
/*! A given set of reference points is given around which the PCF is computed and averaged in a sea of data points.
    Computing the PCF results in a pcf array listing the value of the PCF at each given x, y, z listed in the x, y, and z arrays.

    The values of x, y, z to compute the pcf at are controlled by the xmax, ymax, zmax and n_bins_x, n_bins_y, n_bins_z parameters to the constructor.
    xmax, ymax, zmax determines the minimum/maximum x, y, z at which to compute the pcf and n_bins_x, n_bins_y, n_bins_z is the number of bins in x, y, z.

    <b>2D:</b><br>
    This PCF works for 3D boxes (while it will work for 2D boxes, you should use the 2D version).
*/
class PMFTXYZ
    {
    public:
        //! Constructor
        PMFTXYZ(float max_x, float max_y, float max_z, unsigned int n_bins_x, unsigned int n_bins_y, unsigned int n_bins_z, vec3<float> shiftvec);

        //! Destructor
        ~PMFTXYZ();

        //! Get the simulation box
        const box::Box& getBox() const
            {
            return m_box;
            }

        //! Reset the PCF array to all zeros
        void resetPCF();

        /*! Compute the PCF for the passed in set of points. The function will be added to previous values
            of the pcf
        */
        void accumulate(box::Box& box,
                        const locality::NeighborList *nlist,
                        vec3<float> *ref_points,
                        quat<float> *ref_orientations,
                        unsigned int n_ref,
                        vec3<float> *points,
                        quat<float> *orientations,
                        unsigned int n_p,
                        quat<float> *face_orientations,
                        unsigned int n_faces);

        //! \internal
        //! helper function to reduce the thread specific arrays into one array
        void reducePCF();

        //! Get a reference to the PCF array
        std::shared_ptr<float> getPCF();

        //! Get a reference to the bin counts array
        std::shared_ptr<unsigned int> getBinCounts();

        //! Get a reference to the x array
        std::shared_ptr<float> getX()
            {
            return m_x_array;
            }

        //! Get a reference to the y array
        std::shared_ptr<float> getY()
            {
            return m_y_array;
            }

        //! Get a reference to the z array
        std::shared_ptr<float> getZ()
            {
            return m_z_array;
            }

        float getJacobian()
            {
            return m_jacobian;
            }

        float getRCut()
            {
            return m_r_cut;
            }

        unsigned int getNBinsX()
            {
            return m_n_bins_x;
            }

        unsigned int getNBinsY()
            {
            return m_n_bins_y;
            }

        unsigned int getNBinsZ()
            {
            return m_n_bins_z;
            }

    private:
        box::Box m_box;                    //!< Simulation box where the particles belong
        float m_max_x;                     //!< Maximum x at which to compute pcf
        float m_max_y;                     //!< Maximum y at which to compute pcf
        float m_max_z;                     //!< Maximum z at which to compute pcf
        float m_dx;                        //!< Step size for x in the computation
        float m_dy;                        //!< Step size for y in the computation
        float m_dz;                        //!< Step size for z in the computation
        unsigned int m_n_bins_x;           //!< Number of x bins to compute pcf over
        unsigned int m_n_bins_y;           //!< Number of y bins to compute pcf over
        unsigned int m_n_bins_z;           //!< Number of z bins to compute pcf over
        float m_r_cut;                     //!< r_cut used in cell list construction
        unsigned int m_frame_counter;      //!< number of frames calc'd
        unsigned int m_n_ref;
        unsigned int m_n_p;
        unsigned int m_n_faces;
        float m_jacobian;
        bool m_reduce;
        vec3<float> m_shiftvec;            //!< vector that points from [0,0,0] to the origin of the pmft

        std::shared_ptr<float> m_pcf_array;            //!< array of pcf computed
        std::shared_ptr<unsigned int> m_bin_counts;    //!< array of pcf computed
        std::shared_ptr<float> m_x_array;              //!< array of x values that the pcf is computed at
        std::shared_ptr<float> m_y_array;              //!< array of y values that the pcf is computed at
        std::shared_ptr<float> m_z_array;              //!< array of z values that the pcf is computed at
        tbb::enumerable_thread_specific<unsigned int *> m_local_bin_counts;
    };

}; }; // end namespace freud::pmft

#endif // _PMFTXYZ_H__
