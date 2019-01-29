// Copyright (c) 2010-2019 The Regents of the University of Michigan
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

/*! \file PMFTXYZ.h
    \brief Routines for computing 3D potential of mean force in XYZ coordinates
*/

namespace freud { namespace pmft {

class PMFTXYZ : public PMFT
    {
    public:
        //! Constructor
        PMFTXYZ(float x_max, float y_max, float z_max,
                unsigned int n_x, unsigned int n_y, unsigned int n_z,
                vec3<float> shiftvec);

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
            return m_n_x;
            }

        unsigned int getNBinsY()
            {
            return m_n_y;
            }

        unsigned int getNBinsZ()
            {
            return m_n_z;
            }

    private:
        float m_x_max;                     //!< Maximum x at which to compute pcf
        float m_y_max;                     //!< Maximum y at which to compute pcf
        float m_z_max;                     //!< Maximum z at which to compute pcf
        float m_dx;                        //!< Bin size for x in the computation
        float m_dy;                        //!< Bin size for y in the computation
        float m_dz;                        //!< Bin size for z in the computation
        unsigned int m_n_x;                //!< Number of x bins to compute pcf over
        unsigned int m_n_y;                //!< Number of y bins to compute pcf over
        unsigned int m_n_z;                //!< Number of z bins to compute pcf over
        unsigned int m_n_faces;
        float m_jacobian;
        vec3<float> m_shiftvec;            //!< vector that points from [0,0,0] to the origin of the pmft

        std::shared_ptr<float> m_x_array;              //!< array of x values that the pcf is computed at
        std::shared_ptr<float> m_y_array;              //!< array of y values that the pcf is computed at
        std::shared_ptr<float> m_z_array;              //!< array of z values that the pcf is computed at
    };

}; }; // end namespace freud::pmft

#endif // PMFTXYZ_H
