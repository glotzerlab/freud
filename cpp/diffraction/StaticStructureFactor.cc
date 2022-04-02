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
    // Validation logic is not shared in the parent StaticStructureFactor
    // because StaticStructureFactorDebye can provide a negative k_min to this
    // class's constructor. The k_min value to that class corresponds to the
    // lowest bin center, not the lowest bin's lower edge.

    // Construct the Histogram object that will be used to track the structure factor
    const auto axes = util::Axes {std::make_shared<util::RegularAxis>(bins, k_min, k_max)};
    m_structure_factor = StructureFactorHistogram(axes);
    m_local_structure_factor = StructureFactorHistogram::ThreadLocalHistogram(m_structure_factor);
}

}; }; // namespace freud::diffraction
