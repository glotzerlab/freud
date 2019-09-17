// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef PMFTXYZ_H
#define PMFTXYZ_H

#include "PMFT.h"

/*! \file PMFTXYZ.h
    \brief Routines for computing 3D potential of mean force in XYZ coordinates
*/

namespace freud { namespace pmft {

class PMFTXYZ : public PMFT
{
public:
    //! Constructor
    PMFTXYZ(float x_max, float y_max, float z_max, unsigned int n_x, unsigned int n_y, unsigned int n_z,
            vec3<float> shiftvec);

    /*! Compute the PCF for the passed in set of points. The function will be added to previous values
        of the pcf
    */
    void accumulate(const locality::NeighborQuery* neighbor_query, 
                    quat<float>* orientations, vec3<float>* query_points,
                    unsigned int n_query_points, quat<float>* face_orientations,
                    unsigned int n_faces, const locality::NeighborList* nlist,
                    freud::locality::QueryArgs qargs);

    //! \internal
    //! helper function to reduce the thread specific arrays into one array
    virtual void reducePCF();

    //! Get a reference to the x array
    const util::ManagedArray<float> &getX()
    {
        return m_x_array;
    }

    //! Get a reference to the y array
    const util::ManagedArray<float> &getY()
    {
        return m_y_array;
    }

    //! Get a reference to the z array
    const util::ManagedArray<float> &getZ()
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
    float m_x_max;      //!< Maximum x at which to compute pcf
    float m_y_max;      //!< Maximum y at which to compute pcf
    float m_z_max;      //!< Maximum z at which to compute pcf
    float m_dx;         //!< Bin size for x in the computation
    float m_dy;         //!< Bin size for y in the computation
    float m_dz;         //!< Bin size for z in the computation
    unsigned int m_n_x; //!< Number of x bins to compute pcf over
    unsigned int m_n_y; //!< Number of y bins to compute pcf over
    unsigned int m_n_z; //!< Number of z bins to compute pcf over
    float m_jacobian;
    vec3<float> m_shiftvec; //!< vector that points from [0,0,0] to the origin of the pmft

    util::ManagedArray<float> m_x_array; //!< array of x values that the pcf is computed at
    util::ManagedArray<float> m_y_array; //!< array of y values that the pcf is computed at
    util::ManagedArray<float> m_z_array; //!< array of z values that the pcf is computed at
};

}; }; // end namespace freud::pmft

#endif // PMFTXYZ_H
