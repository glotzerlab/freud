// Copyright (c) 2010-2018 The Regents of the University of Michigan
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
    \brief Routines for computing PMFT in the R12 coordinate system
*/

namespace freud { namespace pmft {

//! Computes the PCF for a given set of points
/*! A given set of reference points is given around which the PCF is computed and averaged in a sea of data points.
    Computing the PCF results in a pcf array listing the value of the PCF at each given x, y, z listed in the x, y, and z arrays.

    The values of r, T1, T2 to compute the pcf at are controlled by the rmax, T1max, T2max and nbins_r, nbins_t1, nbins_t2 parameters to the constructor.
    rmax, T1max, T2max determines the minimum/maximum r, T1, T2 at which to compute the pcf and nbins_r, nbins_t1, nbins_t2 is the number of bins in r, T1, T2.

    <b>2D:</b><br>
    This PCF works for 3D boxes (while it will work for 2D boxes, you should use the 2D version).
*/
class PMFTR12 : public PMFT
    {
    public:
        //! Constructor
        PMFTR12(float max_r, unsigned int nbins_r, unsigned int nbins_t1, unsigned int nbins_t2);

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
            return m_nbins_r;
            }

        unsigned int getNBinsT1()
            {
            return m_nbins_t1;
            }

        unsigned int getNBinsT2()
            {
            return m_nbins_t2;
            }

    private:
        float m_max_r;                     //!< Maximum x at which to compute pcf
        float m_max_t1;                    //!< Maximum y at which to compute pcf
        float m_max_t2;                    //!< Maximum T at which to compute pcf
        float m_dr;                        //!< Step size for x in the computation
        float m_dt1;                       //!< Step size for y in the computation
        float m_dt2;                       //!< Step size for T in the computation
        unsigned int m_nbins_r;            //!< Number of x bins to compute pcf over
        unsigned int m_nbins_t1;           //!< Number of y bins to compute pcf over
        unsigned int m_nbins_t2;           //!< Number of T bins to compute pcf over

        std::shared_ptr<float> m_r_array;              //!< array of x values that the pcf is computed at
        std::shared_ptr<float> m_t1_array;             //!< array of y values that the pcf is computed at
        std::shared_ptr<float> m_t2_array;             //!< array of T values that the pcf is computed at
        std::shared_ptr<float> m_inv_jacobian_array;
    };

}; }; // end namespace freud::pmft

#endif // PMFTR12_H
