// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef PMFTXY2D_H
#define PMFTXY2D_H

#include "PMFT.h"

/*! \file PMFTXY2D.h
    \brief Routines for computing 2D potential of mean force in XY coordinates
*/

namespace freud { namespace pmft {

class PMFTXY2D : public PMFT
{
public:
    //! Constructor
    PMFTXY2D(float x_max, float y_max, unsigned int n_x, unsigned int n_y);

    /*! Compute the PCF for the passed in set of points. The result will
     *  be added to previous values of the PCF.
     */
    void accumulate(const locality::NeighborQuery* neighbor_query, 
                    float* orientations, vec3<float>* query_points,
                    unsigned int n_query_points, 
                    const locality::NeighborList* nlist, freud::locality::QueryArgs qargs);

    //! \internal
    //! helper function to reduce the thread specific arrays into one array
    virtual void reducePCF();

private:
    float m_jacobian;   //!< Determinant of Jacobian, bin area
};

}; }; // end namespace freud::pmft

#endif // PMFTXY2D_H
