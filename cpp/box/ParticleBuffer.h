// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef PARTICLE_BUFFER_H
#define PARTICLE_BUFFER_H

#include <memory>
#include <vector>

#include "Box.h"
#include "VectorMath.h"
#include "Index1D.h"

/*! \file ParticleBuffer.h
    \brief Replicates particles across periodic boundaries.
*/

namespace freud { namespace box {

class ParticleBuffer
    {
    public:
        //! Constructor
        ParticleBuffer(const Box& box) : m_box(box), m_buffer_box(box)
            {
            }

        //! Get the simulation box
        const Box& getBox() const
            {
            return m_box;
            }

        //! Get the buffer box
        const Box& getBufferBox() const
            {
            return m_buffer_box;
            }

        //! Compute the particle images
        void compute(const vec3<float> *points,
                     const unsigned int Np,
                     const float buff,
                     const bool images);

        std::shared_ptr< std::vector< vec3<float> > > getBufferParticles()
            {
            return m_buffer_particles;
            }

        std::shared_ptr< std::vector< unsigned int > > getBufferIds()
            {
            return m_buffer_ids;
            }

    private:
        const Box m_box;    //!< Simulation box of the original particles
        Box m_buffer_box;   //!< Simulation box of the replicated particles
        std::shared_ptr< std::vector< vec3<float> > > m_buffer_particles;
        std::shared_ptr< std::vector< unsigned int > > m_buffer_ids;
    };

}; }; // end namespace freud::box

#endif // PARTICLE_BUFFER_H
