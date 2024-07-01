// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef PERIODIC_BUFFER_H
#define PERIODIC_BUFFER_H

#include <array>
#include <vector>

#include "Box.h"
#include "NeighborQuery.h"
#include "VectorMath.h"

/*! \file PeriodicBuffer.h
    \brief Replicates points across periodic boundaries.
*/

namespace freud { namespace locality {

class PeriodicBuffer
{
public:
    //! Constructor
    PeriodicBuffer() = default;

    //! Get the simulation box
    const freud::box::Box& getBox() const
    {
        return m_box;
    }

    //! Get the buffer box
    const freud::box::Box& getBufferBox() const
    {
        return m_buffer_box;
    }

    //! Compute the periodic buffer
    void compute(std::shared_ptr<NeighborQuery> neighbor_query, std::array<float, 3> buff_vec,
                 const bool use_images, const bool include_input_points);

    //! Return the buffer points
    std::shared_ptr<std::vector<vec3<float>>> getBufferPoints() const
    {
        return m_buffer_points;
    }

    //! Return the buffer ids
    std::shared_ptr<std::vector<unsigned int>> getBufferIds() const
    {
        return m_buffer_ids;
    }

private:
    freud::box::Box m_box;                    //!< Simulation box of the original points
    freud::box::Box m_buffer_box;             //!< Simulation box of the replicated points
    std::shared_ptr<std::vector<vec3<float>>> m_buffer_points; //!< The replicated points
    std::shared_ptr<std::vector<unsigned int>> m_buffer_ids;   //!< The replicated points' original point ids
};

}; }; // end namespace freud::locality

#endif // PERIODIC_BUFFER_H
