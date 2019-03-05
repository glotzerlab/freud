// Copyright (c) 2010-2019 The Regents of the University of Michigan
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
    \brief Routines for computing potential of mean force and torque in XYT coordinates
*/

namespace freud { namespace pmft {

class PMFTXYT : public PMFT
    {
    public:
        //! Constructor
        PMFTXYT(float x_max, float y_max, unsigned int n_x, unsigned int n_y, unsigned int n_t);

        //! Reset the PCF array to all zeros
        virtual void reset();

        /*! Compute the PCF for the passed in set of points. The function will be added to previous values
            of the PCF
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
            return m_n_x;
            }

        unsigned int getNBinsY()
            {
            return m_n_y;
            }

        unsigned int getNBinsT()
            {
            return m_n_t;
            }

    private:
        float m_x_max;                     //!< Maximum x at which to compute PCF
        float m_y_max;                     //!< Maximum y at which to compute PCF
        float m_t_max;                     //!< Maximum t at which to compute PCF
        float m_dx;                        //!< Bin size for x in the computation
        float m_dy;                        //!< Bin size for y in the computation
        float m_dt;                        //!< Bin size for t in the computation
        unsigned int m_n_x;                //!< Number of x bins to compute PCF over
        unsigned int m_n_y;                //!< Number of y bins to compute PCF over
        unsigned int m_n_t;                //!< Number of t bins to compute PCF over
        float m_jacobian;

        std::shared_ptr<float> m_x_array;  //!< Array of x values where the PCF is computed
        std::shared_ptr<float> m_y_array;  //!< Array of y values where the PCF is computed
        std::shared_ptr<float> m_t_array;  //!< Array of t values where the PCF is computed
    };

}; }; // end namespace freud::pmft

#endif // PMFTXYT_H
