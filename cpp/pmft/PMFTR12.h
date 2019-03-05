// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef PMFTR12_H
#define PMFTR12_H

#include <memory>
#include <ostream>
#include <tbb/tbb.h>

#include "Box.h"
#include "VectorMath.h"
#include "LinkCell.h"
#include "PMFT.h"

/*! \file PMFTR12.h
    \brief Routines for computing potential of mean force and torque in R12 coordinates
*/

namespace freud { namespace pmft {

class PMFTR12 : public PMFT
    {
    public:
        //! Constructor
        PMFTR12(float r_max, unsigned int n_r, unsigned int n_t1, unsigned int n_t2);

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

        //! Get a reference to the R array
        std::shared_ptr<float> getR()
            {
            return m_r_array;
            }

        //! Get a reference to the T1 array
        std::shared_ptr<float> getT1()
            {
            return m_t1_array;
            }

        //! Get a reference to the T2 array
        std::shared_ptr<float> getT2()
            {
            return m_t2_array;
            }

        //! Get a reference to the jacobian array
        std::shared_ptr<float> getInverseJacobian()
            {
            return m_inv_jacobian_array;
            }

        unsigned int getNBinsR()
            {
            return m_n_r;
            }

        unsigned int getNBinsT1()
            {
            return m_n_t1;
            }

        unsigned int getNBinsT2()
            {
            return m_n_t2;
            }

    private:
        float m_r_max;                     //!< Maximum r  at which to compute PCF
        float m_t1_max;                    //!< Maximum t1 at which to compute PCF
        float m_t2_max;                    //!< Maximum t2 at which to compute PCF
        float m_dr;                        //!< Bin size for r  in the computation
        float m_dt1;                       //!< Bin size for t1 in the computation
        float m_dt2;                       //!< Bin size for t2 in the computation
        unsigned int m_n_r;                //!< Number of r  bins to compute PCF over
        unsigned int m_n_t1;               //!< Number of t1 bins to compute PCF over
        unsigned int m_n_t2;               //!< Number of t2 bins to compute PCF over

        std::shared_ptr<float> m_r_array;              //!< Array of r  values where the PCF is computed
        std::shared_ptr<float> m_t1_array;             //!< Array of t1 values where the PCF is computed
        std::shared_ptr<float> m_t2_array;             //!< Array of t2 values where the PCF is computed
        std::shared_ptr<float> m_inv_jacobian_array;   //!< Array of inverse jacobians for each bin
    };

}; }; // end namespace freud::pmft

#endif // PMFTR12_H
