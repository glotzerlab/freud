// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is part of the freud project, released under the BSD 3-Clause License.

#include <tbb/tbb.h>
#include <ostream>

// work around nasty issue where python #defines isalpha, toupper, etc....
#undef __APPLE__
#include <Python.h>
#define __APPLE__

#include <memory>

#include "HOOMDMath.h"
#include "VectorMath.h"

#include "LinkCell.h"
#include "box.h"

#ifndef _PMFTR12_H__
#define _PMFTR12_H__

/*! \file PMFTR12.h
    \brief Routines for computing radial density functions
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
class PMFTR12
    {
    public:
        //! Constructor
        PMFTR12(float max_r, unsigned int nbins_r, unsigned int nbins_t1, unsigned int nbins_t2);

        //! Destructor
        ~PMFTR12();

        //! Get the simulation box
        const box::Box& getBox() const
            {
            return m_box;
            }

        //! Reset the PCF array to all zeros
        void resetPCF();

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
        void reducePCF();

        //! Get a reference to the raw bin counts
        std::shared_ptr<unsigned int> getBinCounts();

        //! Get a reference to the PCF array
        std::shared_ptr<float> getPCF();

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

        float getRCut()
            {
            return m_r_cut;
            }

    private:
        box::Box m_box;                    //!< Simulation box where the particles belong
        float m_max_r;                     //!< Maximum x at which to compute pcf
        float m_max_t1;                    //!< Maximum y at which to compute pcf
        float m_max_t2;                    //!< Maximum T at which to compute pcf
        float m_dr;                        //!< Step size for x in the computation
        float m_dt1;                       //!< Step size for y in the computation
        float m_dt2;                       //!< Step size for T in the computation
        unsigned int m_nbins_r;            //!< Number of x bins to compute pcf over
        unsigned int m_nbins_t1;           //!< Number of y bins to compute pcf over
        unsigned int m_nbins_t2;           //!< Number of T bins to compute pcf over
        unsigned int m_frame_counter;      //!< number of frames calc'd
        unsigned int m_n_ref;
        unsigned int m_n_p;
        bool m_reduce;
        float m_r_cut;

        std::shared_ptr<float> m_pcf_array;            //!< array of pcf computed
        std::shared_ptr<unsigned int> m_bin_counts;    //!< array of pcf computed
        std::shared_ptr<float> m_r_array;              //!< array of x values that the pcf is computed at
        std::shared_ptr<float> m_t1_array;             //!< array of y values that the pcf is computed at
        std::shared_ptr<float> m_t2_array;             //!< array of T values that the pcf is computed at
        std::shared_ptr<float> m_inv_jacobian_array;
        tbb::enumerable_thread_specific<unsigned int *> m_local_bin_counts;
    };

}; }; // end namespace freud::pmft

#endif // _PMFTR12_H__
