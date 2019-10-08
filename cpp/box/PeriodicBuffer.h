// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef PERIODIC_BUFFER_H
#define PERIODIC_BUFFER_H

#include <vector>

#include "Box.h"
#include "NeighborQuery.h"
#include "VectorMath.h"

/*! \file PeriodicBuffer.h
    \brief Replicates points across periodic boundaries.
*/

namespace freud { namespace box {

class PeriodicBuffer
{
public:
    //! Constructor
    PeriodicBuffer() {}

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
    void compute(const freud::locality::NeighborQuery* neighbor_query,
                 const vec3<float> buff, const bool use_images);

    //! Return the buffer points
    std::vector<vec3<float> > getBufferPoints() const
    {
        return m_buffer_points;
    }

    //! Return the buffer ids
    std::vector<unsigned int> getBufferIds() const
    {
        return m_buffer_ids;
    }

private:
    Box m_box;  //!< Simulation box of the original points
    Box m_buffer_box; //!< Simulation box of the replicated points
    std::vector<vec3<float> > m_buffer_points; //!< The replicated points
    std::vector<unsigned int> m_buffer_ids; //!< The replicated points' original point ids
};

}; }; // end namespace freud::box

#endif // PERIODIC_BUFFER_H
