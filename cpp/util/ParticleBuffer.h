// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef VORONOIBUFFER_H
#define VORONOIBUFFER_H

#include <memory>
#include <vector>

#include "box.h"
#include "VectorMath.h"
#include "Index1D.h"

/*! \file ParticleBuffer.h
    \brief Computes a buffer of particles to support wrapped positions in qhull
*/

namespace freud { namespace util {

//! Locates the particles near the border of the box and computes their nearest images to pass to qhull
class ParticleBuffer
    {
    public:
        //! Constructor
        ParticleBuffer(const box::Box& box):m_box(box){}

        //! Get the simulation box
        const box::Box& getBox() const
                {
                return m_box;
                }

        //! Compute the particle images
        void compute(const vec3<float> *points,
                     const unsigned int Np,
                     const float buff);

        std::shared_ptr< std::vector< vec3<float> > > getBufferParticles()
            {
            return m_buffer_particles;
            }

        std::shared_ptr< std::vector< unsigned int > > getBufferIds()
            {
            return m_buffer_ids;
            }

    private:
        const box::Box m_box;    //!< Simulation box where the particles belong
        std::shared_ptr< std::vector< vec3<float> > > m_buffer_particles;
        std::shared_ptr< std::vector< unsigned int > > m_buffer_ids;
    };

}; }; // end namespace freud::util

#endif // VORONOI_BUFFER_H
