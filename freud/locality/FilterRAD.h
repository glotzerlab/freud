// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#pragma once

#include "Filter.h"

#include "NeighborList.h"
#include "NeighborQuery.h"
#include <memory>
#include "VectorMath.h"

namespace freud { namespace locality {

/* Class for the RAD method of filtering a neighborlist.
 * */
class FilterRAD : public Filter
{
public:
    //<! Construct with an empty NeighborList, fill it upon calling compute
    FilterRAD(bool allow_incomplete_shell, bool terminate_after_blocked)
        : Filter(allow_incomplete_shell), m_terminate_after_blocked(terminate_after_blocked)
    {}

    void compute(std::shared_ptr<NeighborQuery> nq, const vec3<float>* query_points,
                 unsigned int num_query_points, std::shared_ptr<NeighborList> nlist,
                 const QueryArgs& qargs) override;

private:
    //<! variable that determines if RAD-open (False) or RAD-closed (True) is computed
    bool m_terminate_after_blocked;
};

}; }; // namespace freud::locality
