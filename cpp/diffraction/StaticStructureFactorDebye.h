// Copyright (c) 2010-2023 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef STATIC_STRUCTURE_FACTOR_DEBYE_H
#define STATIC_STRUCTURE_FACTOR_DEBYE_H

#include <limits>

#include "Histogram.h"
#include "NeighborQuery.h"
#include "StaticStructureFactor.h"

/*! \file StaticStructureFactorDebye.h
    \brief Computes structure factor using the Debye scattering equation.

    This method is not capable of resolving k-vectors smaller than the magnitude 4 * pi / L, where L is the
    smallest side length of the system's periodic box.
*/

namespace freud { namespace diffraction {

class StaticStructureFactorDebye : public StaticStructureFactor
{
public:
    //! Constructor
    StaticStructureFactorDebye(unsigned int bins, float k_max, float k_min = 0);

    //! Compute the structure factor S(k) using the Debye formula
    void accumulate(const freud::locality::NeighborQuery* neighbor_query, const vec3<float>* query_points,
                    unsigned int n_query_points, unsigned int n_total) override;

    //! Reset the histogram to all zeros
    void reset() override
    {
        m_local_structure_factor.reset();
        m_frame_counter = 0;
        m_min_valid_k = std::numeric_limits<float>::infinity();
        m_reduce = true;
    }

private:
    //! Reduce thread-local arrays onto the primary data arrays.
    void reduce() override;

    unsigned int m_frame_counter {0}; //!< Number of frames calculated
};

}; }; // namespace freud::diffraction

#endif // STATIC_STRUCTURE_FACTOR_DEBYE_H
