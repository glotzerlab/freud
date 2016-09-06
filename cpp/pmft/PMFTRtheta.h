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
#include "Index1D.h"

#ifndef _PMFTRtheta_H__
#define _PMFTRtheta_H__

/*! \internal
    \file PMFTRtheta.h
    \brief Routines for computing anisotropic potential of mean force in 3D
*/

namespace freud { namespace pmft {

//! Computes the PCF for a given set of points
/*! A given set of reference points is given around which the PCF is computed and averaged in a sea of data points.
    Computing the PCF results in a pcf array listing the value of the PCF at each given R and theta listed.
    R is just the 3D radial separation, and theta is the magnitude of the angle that separates the reference orientation vs. the neighbor orientation.
    Theta does not contain information about the type of angular deviation. This is meant to be a first step and we can modify things to capture the different scalar invariants that take into account the type of deviation from two particle's orientational deviation.

    The values of R, theta to compute the pcf at are controlled by the Rmax, theta_max and n_bins_R, n_bins_theta parameters to the constructor.
    Rmax, theta_max determines the minimum/maximum R and theta at which to compute the pcf and n_bins_R, n_bins_theta is the number of bins in R, theta.

    <b>2D:</b><br>
    This PCF works for 3D boxes (while it will work for 2D boxes, you should use the 2D version).
*/
class PMFTRtheta
    {
    public:
        //! Constructor
        PMFTRtheta(float max_R, float max_theta, unsigned int n_bins_R, unsigned int n_bins_theta);

        //! Destructor
        ~PMFTRtheta();

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
                        vec3<float> *ref_points,
                        quat<float> *ref_orientations,
                        unsigned int n_ref,
                        vec3<float> *points,
                        quat<float> *orientations,
                        unsigned int n_p,
                        quat<float> *equivalent_orientations,
                        unsigned int n_q);

        //! \internal
        //! helper function to reduce the thread specific arrays into the boost array
        void reducePCF();

        //! Get a reference to the PCF array
        std::shared_ptr<float> getPCF();

        //! Get a reference to the bin counts array
        std::shared_ptr<unsigned int> getBinCounts();

        //! Get a reference to the R array
        std::shared_ptr<float> getR()
            {
            return m_R_array;
            }

        //! Get a reference to the theta array
        std::shared_ptr<float> get_theta()
            {
            return m_theta_array;
            }


        float getJacobian()
            {
            return m_jacobian;
            }

        //! Get a reference to the jacobian array
        std::shared_ptr<float> getInverseJacobian()
            {
            return m_inv_jacobian_array;
            }

        float getRCut()
            {
            return m_r_cut;
            }

        unsigned int getNBinsR()
            {
            return m_n_bins_R;
            }

        unsigned int getNBins_theta()
            {
            return m_n_bins_theta;
            }


    private:
        box::Box m_box;            //!< Simulation box the particles belong in
        float m_max_R;                     //!< Maximum R at which to compute pcf
        float m_max_theta;                     //!< Maximum theta at which to compute pcf
        float m_dR;                       //!< Step size for R in the computation
        float m_d_theta;                       //!< Step size for theta in the computation
        locality::LinkCell* m_lc;          //!< LinkCell to bin particles for the computation
        unsigned int m_n_bins_R;             //!< Number of R bins to compute pcf over
        unsigned int m_n_bins_theta;             //!< Number of theta bins to compute pcf over
        float m_r_cut;                      //!< r_cut used in cell list construction
        unsigned int m_frame_counter;       //!< number of frames calc'd
        unsigned int m_n_ref;
        unsigned int m_n_p;
        unsigned int m_n_q;
        float m_jacobian;
        bool m_reduce;

        std::shared_ptr<float> m_pcf_array;         //!< array of pcf computed
        std::shared_ptr<unsigned int> m_bin_counts;         //!< array of pcf computed
        std::shared_ptr<float> m_R_array;           //!< array of R values that the pcf is computed at
        std::shared_ptr<float> m_theta_array;           //!< array of theta values that the pcf is computed at
        std::shared_ptr<float> m_inv_jacobian_array;
        tbb::enumerable_thread_specific<unsigned int *> m_local_bin_counts;
    };

}; }; // end namespace freud::pmft

#endif // _PMFTRtheta_H__
