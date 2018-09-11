// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef PMFTXYT_H
#define PMFTXYT_H

#include <memory>
#include <ostream>
#include <tbb/tbb.h>

#include "Box.h"
#include "VectorMath.h"
#include "LinkCell.h"
#include "PMFT.h"

/*! \file PMFTXYT.h
    \brief Routines for computing PMFT in the XYT coordinate system
*/

namespace freud { namespace pmft {

class PMFTXYT : public PMFT
    {
    public:
        //! Constructor
        PMFTXYT(float max_x, float max_y, unsigned int n_bins_x, unsigned int n_bins_y, unsigned int n_bins_t);

        //! Reset the PCF array to all zeros
        virtual void reset();

        /*! Compute the PCF for the passed in set of points. The function will be added to previous values
            of the pcf
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

        //! Get a reference to the X array
        std::shared_ptr<float> getX()
            {
            return m_x_array;
            }

        //! Get a reference to the Y array
        std::shared_ptr<float> getY()
            {
            return m_y_array;
            }

        //! Get a reference to the T array
        std::shared_ptr<float> getT()
            {
            return m_t_array;
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

        unsigned int getNBinsT()
            {
            return m_n_bins_t;
            }

    private:
        float m_max_x;                     //!< Maximum x at which to compute pcf
        float m_max_y;                     //!< Maximum y at which to compute pcf
        float m_max_t;                     //!< Maximum T at which to compute pcf
        float m_dx;                        //!< Step size for x in the computation
        float m_dy;                        //!< Step size for y in the computation
        float m_dt;                        //!< Step size for T in the computation
        unsigned int m_n_bins_x;           //!< Number of x bins to compute pcf over
        unsigned int m_n_bins_y;           //!< Number of y bins to compute pcf over
        unsigned int m_n_bins_t;           //!< Number of T bins to compute pcf over
        float m_jacobian;

        std::shared_ptr<float> m_x_array;              //!< array of x values that the pcf is computed at
        std::shared_ptr<float> m_y_array;              //!< array of y values that the pcf is computed at
        std::shared_ptr<float> m_t_array;              //!< array of T values that the pcf is computed at
    };

}; }; // end namespace freud::pmft

#endif // PMFTXYT_H
