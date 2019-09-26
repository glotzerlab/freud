// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef PERIODIC_BUFFER_H
#define PERIODIC_BUFFER_H

#include <vector>

#include "Box.h"
#include "VectorMath.h"

/*! \file PeriodicBuffer.h
    \brief Replicates points across periodic boundaries.
*/

namespace freud { namespace box {

class PeriodicBuffer
{
public:
    //! Constructor
    PeriodicBuffer(const Box& box) : m_box(box), m_buffer_box(box) {}

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

    //! Compute the periodic buffer
    void compute(const vec3<float>* points, const unsigned int Np, const vec3<float> buff,
                 const bool use_images);

    std::vector<vec3<float> > getBufferPoints()
    {
        return m_buffer_points;
    }

    std::vector<unsigned int> getBufferIds()
    {
        return m_buffer_ids;
    }

private:
    const Box m_box;  //!< Simulation box of the original points
    Box m_buffer_box; //!< Simulation box of the replicated points
    std::vector<vec3<float> > m_buffer_points;
    std::vector<unsigned int> m_buffer_ids;
};

}; }; // end namespace freud::box

#endif // PERIODIC_BUFFER_H
