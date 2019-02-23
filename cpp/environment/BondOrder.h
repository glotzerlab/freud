// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef BOND_ORDER_H
#define BOND_ORDER_H

#include <memory>
#include <ostream>
#include <tbb/tbb.h>

#include "Box.h"
#include "VectorMath.h"
#include "NearestNeighbors.h"
#include "Index1D.h"

/*! \file BondOrder.h
    \brief Compute the bond order diagram for the system of particles.
*/

namespace freud { namespace environment {

// this is needed for conversion of the type of bond order calculation to be made in accumulate.
typedef enum {bod=0, lbod=1, obcd=2, oocd=3} BondOrderMode;

//! Compute the bond order parameter for a set of points
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
        void reset();

        //! Accumulate the bond order
        void accumulate(box::Box& box,
                        const freud::locality::NeighborList *nlist,
                        vec3<float> *ref_points,
                        quat<float> *ref_orientations,
                        unsigned int n_ref,
                        vec3<float> *points,
                        quat<float> *orientations,
                        unsigned int n_p,
                        unsigned int mode);

        void reduceBondOrder();

        //! Get a reference to the last computed bond order
        std::shared_ptr<float> getBondOrder();

        //! Get a reference to the theta array
        std::shared_ptr<float> getTheta()
            {
            return m_theta_array;
            }

        //! Get a reference to the phi array
        std::shared_ptr<float> getPhi()
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

    private:
        box::Box m_box;                //!< Simulation box where the particles belong
        float m_dt;
        float m_dp;
        unsigned int m_n_ref;          //!< Last number of points computed
        unsigned int m_n_p;            //!< Last number of points computed
        unsigned int m_nbins_t;        //!< number of bins for theta
        unsigned int m_nbins_p;        //!< number of bins for phi
        unsigned int m_frame_counter;  //!< number of frames calc'd
        bool m_reduce;                 //!< Whether arrays need to be reduced across threads

        std::shared_ptr<unsigned int> m_bin_counts;  //!< bin counts computed
        std::shared_ptr<float> m_bo_array;           //!< bond order array computed
        std::shared_ptr<float> m_sa_array;           //!< surface area array computed
        std::shared_ptr<float> m_theta_array;        //!< theta array computed
        std::shared_ptr<float> m_phi_array;          //!< phi order array computed
        tbb::enumerable_thread_specific<unsigned int *> m_local_bin_counts;
    };

}; }; // end namespace freud::environment

#endif // BOND_ORDER_H
