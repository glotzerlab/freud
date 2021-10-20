// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <stdexcept>

#include "Histogram.h"
#include "StaticStructureFactor.h"

/*! \file StaticStructureFactor.cc
    \brief Base class for structure factor classes.
*/

namespace freud { namespace diffraction {

StaticStructureFactor::StaticStructureFactor(unsigned int bins, float k_max, float k_min)
{
    if (bins == 0)
    {
        throw std::invalid_argument("StaticStructureFactor requires a nonzero number of bins.");
    }
    if (k_max <= 0)
    {
        throw std::invalid_argument("StaticStructureFactor requires k_max to be positive.");
    }
    if (k_min < 0)
    {
        throw std::invalid_argument("StaticStructureFactor requires k_min to be non-negative.");
    }
    if (k_max <= k_min)
    {
        throw std::invalid_argument("StaticStructureFactor requires that k_max must be greater than k_min.");
    }
    // Construct the Histogram object that will be used to track the structure factor
    const auto axes
        = StructureFactorHistogram::Axes {std::make_shared<util::RegularAxis>(bins, k_min, k_max)};
    m_structure_factor = StructureFactorHistogram(axes);
    m_local_structure_factor = StructureFactorHistogram::ThreadLocalHistogram(m_structure_factor);
}

}; }; // namespace freud::diffraction
