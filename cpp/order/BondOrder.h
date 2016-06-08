#include <tbb/tbb.h>
#include <ostream>

// work around nasty issue where python #defines isalpha, toupper, etc....
#undef __APPLE__
#include <Python.h>
#define __APPLE__

#include <boost/shared_array.hpp>

#include "HOOMDMath.h"
#include "VectorMath.h"

#include "NearestNeighbors.h"
#include "box.h"
#include "Index1D.h"

#ifndef _BOND_ORDER_H__
#define _BOND_ORDER_H__

/*! \file BondOrder.h
    \brief Compute the hexatic order parameter for each particle
*/

namespace freud { namespace order {

//! Compute the hexagonal order parameter for a set of points
/*!
*/
class BondOrder
    {
    public:
        //! Constructor
        BondOrder(float rmax, float k, unsigned int n, unsigned int nbins_t, unsigned int nbins_p);

        //! Destructor
        ~BondOrder();

        //! Get the simulation box
        const box::Box& getBox() const
            {
            return m_box;
            }

        //! Reset the bond order array to all zeros
        void resetBondOrder();

        //! Python wrapper for reset method
        // void resetBondOrderPy()
        //     {
        //     resetBondOrder();
        //     }

        //! accumulate the bond order
        void accumulate(box::Box& box,
                        vec3<float> *ref_points,
                        quat<float> *ref_orientations,
                        unsigned int Nref,
                        vec3<float> *points,
                        quat<float> *orientations,
                        unsigned int Np);

        // //! Python wrapper for accumulate
        // void accumulatePy(box::Box& box,
        //                   boost::python::numeric::array ref_points,
        //                   boost::python::numeric::array ref_orientations,
        //                   boost::python::numeric::array points,
        //                   boost::python::numeric::array orientations);

        //! Compute the bond order
        // void compute(vec3<float> *ref_points,
        //              quat<float> *ref_orientations,
        //              unsigned int Nref,
        //              vec3<float> *points,
        //              quat<float> *orientations,
        //              unsigned int Np);

        // //! Python wrapper for compute
        // void computePy(box::Box& box,
        //                boost::python::numeric::array ref_points,
        //                boost::python::numeric::array ref_orientations,
        //                boost::python::numeric::array points,
        //                boost::python::numeric::array orientations);

        void reduceBondOrder();

        //! Get a reference to the last computed rdf
        boost::shared_array<float> getBondOrder();

        //! Get a reference to the r array
        boost::shared_array<float> getTheta()
            {
            return m_theta_array;
            }

        //! Get a reference to the N_r array
        boost::shared_array<float> getPhi()
            {
            return m_phi_array;
            }

        unsigned int getNBinsTheta()
            {
            return m_nbins_t;
            }

        unsigned int getNBinsPhi()
            {
            return m_nbins_p;
            }

        // //! Python wrapper for getRDF() (returns a copy)
        // boost::python::numeric::array getBondOrderPy();

        // //! Python wrapper for getR() (returns a copy)
        // boost::python::numeric::array getThetaPy()
        //     {
        //     float *arr = m_theta_array.get();
        //     return num_util::makeNum(arr, m_nbins_t);
        //     }

        // //! Python wrapper for getNr() (returns a copy)
        // boost::python::numeric::array getPhiPy()
        //     {
        //     float *arr = m_phi_array.get();
        //     return num_util::makeNum(arr, m_nbins_p);
        //     }

    private:
        box::Box m_box;            //!< Simulation box the particles belong in
        float m_rmax;                     //!< Maximum r at which to determine neighbors
        float m_k;                        //!< Multiplier in the exponent
        float m_dt;
        float m_dp;
        locality::NearestNeighbors *m_nn;          //!< Nearest Neighbors for the computation
        unsigned int m_Nref;                //!< Last number of points computed
        unsigned int m_Np;                //!< Last number of points computed
        unsigned int m_nbins_t;           //!< number of bins for theta
        unsigned int m_nbins_p;           //!< number of bins for phi
        unsigned int m_frame_counter;       //!< number of frames calc'd

        boost::shared_array<unsigned int> m_bin_counts;         //!< bin counts computed
        boost::shared_array<float> m_bo_array;         //!< bond order array computed
        boost::shared_array<float> m_sa_array;         //!< bond order array computed
        boost::shared_array<float> m_theta_array;         //!< theta array computed
        boost::shared_array<float> m_phi_array;         //!< phi order array computed
        tbb::enumerable_thread_specific<unsigned int *> m_local_bin_counts;
    };

}; }; // end namespace freud::order

#endif // _BOND_ORDER_H__
