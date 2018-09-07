// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef PMFTXY2D_H
#define PMFTXY2D_H

#include <memory>
#include <ostream>
#include <tbb/tbb.h>

#include "Box.h"
#include "VectorMath.h"
#include "LinkCell.h"
#include "Index1D.h"
#include "PMFT.h"

/*! \internal
    \file PMFTXY2D.h
    \brief Routines for computing anisotropic potential of mean force in 2D
*/

namespace freud { namespace pmft {

//! Computes the PCF for a given set of points
/*! A given set of reference points is given around which the PCF is computed and averaged in a sea of data points.
    Computing the PCF results in a pcf array listing the value of the PCF at each given x, y, listed in the x, y arrays.

    The values of x, y to compute the pcf at are controlled by the xmax, ymax and n_bins_x, n_bins_y parameters to the constructor.
    xmax, ymax determines the minimum/maximum x, y at which to compute the pcf and n_bins_x, n_bins_y is the number of bins in x, y.

    <b>2D:</b><br>
    This PCF only works for 2D boxes. As with everything else in freud, 2D points must be passed in as
    3 component vectors x,y,0. Failing to set 0 in the third component should not matter as the code forces z=0.
    However, this could still lead to undefined behavior and should be avoided anyway.
*/
class PMFTXY2D : public PMFT
    {
    public:
        //! Constructor
        PMFTXY2D(float max_x, float max_y, unsigned int n_bins_x, unsigned int n_bins_y);

        //! Reset the PCF array to all zeros
        virtual void reset();

        /*! Compute the PCF for the passed in set of points. The result will
         *  be added to previous values of the PCF.
         */
        void accumulate(box::Box& box,
                        const locality::NeighborList *nlist,
                        vec3<float> *ref_points,
                        float *ref_orientations,
                        unsigned int n_ref,
                        vec3<float> *points,
                        float *orientations,
                        unsigned int n_p);

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

        //! Get the jacobian determinant (not the matrix)
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

    private:
        float m_max_x;                 //!< Maximum x at which to compute pcf
        float m_max_y;                 //!< Maximum y at which to compute pcf
        float m_dx;                    //!< Step size for x in the computation
        float m_dy;                    //!< Step size for y in the computation
        unsigned int m_n_bins_x;       //!< Number of x bins to compute pcf over
        unsigned int m_n_bins_y;       //!< Number of y bins to compute pcf over
        float m_jacobian;

        std::shared_ptr<float> m_x_array;            //!< array of x values where the pcf is computed
        std::shared_ptr<float> m_y_array;            //!< array of y values where the pcf is computed
    };

}; }; // end namespace freud::pmft

#endif // PMFTXY2D_H
