// Copyright (c) 2010-2016 The Regents of the University of Michigan
// This file is part of the Freud project, released under the BSD 3-Clause License.

#include <tbb/tbb.h>
#include <ostream>

// work around nasty issue where python #defines isalpha, toupper, etc....
#undef __APPLE__
#include <Python.h>
#define __APPLE__

#include <memory>

#include "HOOMDMath.h"
#include "VectorMath.h"
#include "TensorMath.h"
#include "saruprng.h"

#include "NearestNeighbors.h"
#include "box.h"
#include "Index1D.h"

#ifndef _NEMATIC_ORDER_PARAMETER_H__
#define _NEMATIC_ORDER_PARAMETER_H__

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

        //! Reset the bond order array to all zeros
        void resetNematicOrderParameter();

        //! accumulate the bond order
        void compute(quat<float> *orientations,
                     unsigned int n);

        //! Get a reference to the last computed rdf
        float getNematicOrderParameter();

        std::shared_ptr<float> getParticleTensor();

        std::shared_ptr<float> getNematicTensor();

        unsigned int getNumParticles();

        vec3<float> getNematicDirector();

    private:
        vec3<float> m_u;                 //!< The molecular axis
        unsigned int m_n;                //!< Last number of points computed

        float m_nematic_order_parameter;
        vec3<float> m_nematic_director;
        float m_nematic_tensor[9];
        std::shared_ptr<float> m_particle_order_parameter;
        std::shared_ptr<float> m_particle_tensor;
    };

}; }; // end namespace freud::order

#endif // _NEMATIC_ORDER_PARAMETER_H__
