// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef NEMATIC_ORDER_PARAMETER_H
#define NEMATIC_ORDER_PARAMETER_H

#include <memory>
#include <ostream>
#include <tbb/tbb.h>

#include "Box.h"
#include "VectorMath.h"
#include "TensorMath.h"
#include "saruprng.h"
#include "NearestNeighbors.h"
#include "Index1D.h"

/*! \file NematicOrderParameter.h
    \brief Compute the nematic order parameter for each particle
*/

namespace freud { namespace order {
//! Compute the nematic order parameter for a set of points
/*!
*/
class NematicOrderParameter
    {
    public:
        //! Constructor
        NematicOrderParameter(vec3<float> u);

        //! Destructor
        virtual ~NematicOrderParameter() {};

        //! Reset the nematic order parameter array to all zeros
        void reset();

        //! Compute the nematic order parameter
        void compute(quat<float> *orientations,
                     unsigned int n);

        //! Get the value of the last computed nematic order parameter
        float getNematicOrderParameter();

        std::shared_ptr<float> getParticleTensor();

        std::shared_ptr<float> getNematicTensor();

        unsigned int getNumParticles();

        vec3<float> getNematicDirector();

    private:
        unsigned int m_n;                //!< Last number of points computed
        vec3<float> m_u;                 //!< The molecular axis
        float m_nematic_order_parameter; //!< Current value of the order parameter
        vec3<float> m_nematic_director;  //!< The director (eigenvector corresponding to the OP)
        float m_nematic_tensor[9];       //!< The Q tensor

        std::shared_ptr<float> m_sp_nematic_tensor; //!< Pointer to nematic tensor that is passed back
                                                    //!< to python to provide a view into the object
        std::shared_ptr<float> m_particle_tensor;   //!< The per-particle tensor that is summed up to Q
                                                    //!< Used to allow parallelized calculation of Q
    };

}; }; // end namespace freud::order

#endif // NEMATIC_ORDER_PARAMETER_H
