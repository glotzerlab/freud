// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef PMFTXYZ_H
#define PMFTXYZ_H

#include <memory>
#include <ostream>
#include <tbb/tbb.h>

#include "Box.h"
#include "VectorMath.h"
#include "LinkCell.h"
#include "Index1D.h"
#include "PMFT.h"

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
class PMFTXYZ : public PMFT
    {
    public:
        //! Constructor
        PMFTXYZ(float max_x, float max_y, float max_z, unsigned int n_bins_x, unsigned int n_bins_y, unsigned int n_bins_z, vec3<float> shiftvec);

        //! Reset the PCF array to all zeros
        virtual void reset();

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
        virtual void reducePCF();

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
        float m_max_x;                     //!< Maximum x at which to compute pcf
        float m_max_y;                     //!< Maximum y at which to compute pcf
        float m_max_z;                     //!< Maximum z at which to compute pcf
        float m_dx;                        //!< Step size for x in the computation
        float m_dy;                        //!< Step size for y in the computation
        float m_dz;                        //!< Step size for z in the computation
        unsigned int m_n_bins_x;           //!< Number of x bins to compute pcf over
        unsigned int m_n_bins_y;           //!< Number of y bins to compute pcf over
        unsigned int m_n_bins_z;           //!< Number of z bins to compute pcf over
        unsigned int m_n_faces;
        float m_jacobian;
        vec3<float> m_shiftvec;            //!< vector that points from [0,0,0] to the origin of the pmft

        std::shared_ptr<float> m_x_array;              //!< array of x values that the pcf is computed at
        std::shared_ptr<float> m_y_array;              //!< array of y values that the pcf is computed at
        std::shared_ptr<float> m_z_array;              //!< array of z values that the pcf is computed at
    };

}; }; // end namespace freud::pmft

#endif // PMFTXYZ_H
