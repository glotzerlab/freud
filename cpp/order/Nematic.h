// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef NEMATIC_H
#define NEMATIC_H

#include <memory>
#include <tbb/tbb.h>

#include "Box.h"
#include "ManagedArray.h"
#include "VectorMath.h"

/*! \file Nematic.h
    \brief Compute the nematic order parameter for each particle
*/

namespace freud { namespace order {
//! Compute the nematic order parameter for a set of points
/*!
 */
class Nematic
{
public:
    //! Constructor
    Nematic(vec3<float> u);

    //! Destructor
    virtual ~Nematic() {};

    //! Compute the nematic order parameter
    void compute(quat<float>* orientations, unsigned int n);

    //! Get the value of the last computed nematic order parameter
    float getNematicOrderParameter() const;

    const util::ManagedArray<float>& getParticleTensor() const;

    const util::ManagedArray<float>& getNematicTensor() const;

    unsigned int getNumParticles() const;

    vec3<float> getNematicDirector() const;

    vec3<float> getU() const;

private:
    unsigned int m_n;                //!< Last number of points computed
    vec3<float> m_u;                 //!< The molecular axis
    float m_nematic_order_parameter; //!< Current value of the order parameter
    vec3<float> m_nematic_director;  //!< The director (eigenvector corresponding to the OP)

    util::ManagedArray<float> m_nematic_tensor;  //!< The computed nematic tensor.
    util::ManagedArray<float> m_particle_tensor; //!< The per-particle tensor that is summed up to Q. Used to
                                                 //!< allow parallelized calculation of Q
};

}; }; // end namespace freud::order

#endif // NEMATIC_H
