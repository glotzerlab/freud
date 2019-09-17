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

    //! Get the jacobian determinant (not the matrix)
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

private:
    float m_x_max;      //!< Maximum x at which to compute PCF
    float m_y_max;      //!< Maximum y at which to compute PCF
    float m_dx;         //!< Bin size for x in the computation
    float m_dy;         //!< Bin size for y in the computation
    unsigned int m_n_x; //!< Number of x bins to compute PCF over
    unsigned int m_n_y; //!< Number of y bins to compute PCF over
    float m_jacobian;   //!< Determinant of Jacobian, bin area

    util::ManagedArray<float> m_x_array; //!< Array of x values where the PCF is computed
    util::ManagedArray<float> m_y_array; //!< Array of y values where the PCF is computed
};

}; }; // end namespace freud::pmft

#endif // PMFTXY2D_H
