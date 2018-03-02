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

#ifndef _PMFTXYT_H__
#define _PMFTXYT_H__

/*! \file PMFTXYT.h
    \brief Routines for computing radial density functions
*/

namespace freud { namespace pmft {

class PMFTXYT
    {
    public:
        //! Constructor
        PMFTXYT(float max_x, float max_y, unsigned int n_bins_x, unsigned int n_bins_y, unsigned int n_bins_t);

        //! Destructor
        ~PMFTXYT();

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
        std::shared_ptr<float> getX()
            {
            return m_x_array;
            }

        //! Get a reference to the T1 array
        std::shared_ptr<float> getY()
            {
            return m_y_array;
            }

        //! Get a reference to the T2 array
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

        float getRCut()
            {
            return m_r_cut;
            }

    private:
        box::Box m_box;                    //!< Simulation box where the particles belong
        float m_max_x;                     //!< Maximum x at which to compute pcf
        float m_max_y;                     //!< Maximum y at which to compute pcf
        float m_max_t;                     //!< Maximum T at which to compute pcf
        float m_dx;                        //!< Step size for x in the computation
        float m_dy;                        //!< Step size for y in the computation
        float m_dt;                        //!< Step size for T in the computation
        unsigned int m_n_bins_x;           //!< Number of x bins to compute pcf over
        unsigned int m_n_bins_y;           //!< Number of y bins to compute pcf over
        unsigned int m_n_bins_t;           //!< Number of T bins to compute pcf over
        unsigned int m_frame_counter;      //!< number of frames calc'd
        unsigned int m_n_ref;
        unsigned int m_n_p;
        bool m_reduce;
        float m_r_cut;
        float m_jacobian;

        std::shared_ptr<float> m_pcf_array;            //!< array of pcf computed
        std::shared_ptr<unsigned int> m_bin_counts;    //!< array of pcf computed
        std::shared_ptr<float> m_x_array;              //!< array of x values that the pcf is computed at
        std::shared_ptr<float> m_y_array;              //!< array of y values that the pcf is computed at
        std::shared_ptr<float> m_t_array;              //!< array of T values that the pcf is computed at
        tbb::enumerable_thread_specific<unsigned int *> m_local_bin_counts;
    };

}; }; // end namespace freud::pmft

#endif // _PMFTXYT_H__
