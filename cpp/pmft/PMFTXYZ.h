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
    void accumulate(const locality::NeighborQuery* neighbor_query, quat<float>* query_orientations,
                    vec3<float>* query_points, unsigned int n_query_points, quat<float>* face_orientations,
                    unsigned int n_faces, const locality::NeighborList* nlist,
                    freud::locality::QueryArgs qargs);

    //! \internal
    //! helper function to reduce the thread specific arrays into one array
    virtual void reduce();

private:
    float m_jacobian;
    vec3<float> m_shiftvec; //!< vector that points from [0,0,0] to the origin of the pmft
};

}; }; // end namespace freud::pmft

#endif // PMFTXYZ_H
