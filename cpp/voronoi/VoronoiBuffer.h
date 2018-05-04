// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is part of the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <vector>

#include "box.h"
#include "VectorMath.h"
#include "Index1D.h"

#ifndef _VORONOIBUFFER_H__
#define _VORONOIBUFFER_H__

/*! \file VoronoiBuffer.h
    \brief Computes a buffer of particles to support wrapped positions in qhull
*/

namespace freud { namespace voronoi {

//! Locates the particles near the border of the box and computes their nearest images to pass to qhull
class VoronoiBuffer
    {
    public:
        //! Constructor
        VoronoiBuffer(const box::Box& box):m_box(box){}

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

}; }; // end namespace freud::voronoi

#endif
