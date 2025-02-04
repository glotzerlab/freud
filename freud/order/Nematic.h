// Copyright (c) 2010-2025 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef NEMATIC_H
#define NEMATIC_H

#include <memory>

#include "Box.h"
#include "ManagedArray.h"
#include "ThreadStorage.h"
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
    Nematic()
    {
        m_nematic_tensor = std::make_shared<util::ManagedArray<float>>(std::vector<size_t> {3, 3});
        m_nematic_tensor_local = std::make_shared<util::ThreadStorage<float>>(std::vector<size_t> {3, 3});
    }

    //! Destructor
    virtual ~Nematic() = default;

    //! Compute the nematic order parameter
    void compute(vec3<float>* orientations, unsigned int n);

    //! Get the value of the last computed nematic order parameter
    float getNematicOrderParameter() const;

    std::shared_ptr<util::ManagedArray<float>> getParticleTensor() const;

    std::shared_ptr<util::ManagedArray<float>> getNematicTensor() const;

    unsigned int getNumParticles() const;

    vec3<float> getNematicDirector() const;

private:
    unsigned int m_n {0};                //!< Last number of points computed
    float m_nematic_order_parameter {0}; //!< Current value of the order parameter
    vec3<float> m_nematic_director;      //!< The director (eigenvector corresponding to the OP)

    std::shared_ptr<util::ManagedArray<float>> m_nematic_tensor;        //!< The computed nematic tensor.
    std::shared_ptr<util::ThreadStorage<float>> m_nematic_tensor_local; //!< Thread-specific nematic tensor.
    std::shared_ptr<util::ManagedArray<float>>
        m_particle_tensor; //!< The per-particle tensor that is summed up to Q.
};

}; }; // end namespace freud::order

#endif // NEMATIC_H
