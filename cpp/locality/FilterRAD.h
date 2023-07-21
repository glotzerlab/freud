// Copyright (c) 2010-2023 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef __FILTERRAD_H__
#define __FILTERRAD_H__

#include "Filter.h"

#include "NeighborList.h"
#include "NeighborQuery.h"

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

    void compute(const NeighborQuery* nq, const vec3<float>* query_points, unsigned int num_query_points,
                 const NeighborList* nlist, const QueryArgs& qargs) override;

private:
    //<! variable that determines if RAD-open (False) or RAD-closed (True) is computed
    bool m_terminate_after_blocked;
};

}; }; // namespace freud::locality

#endif // __FILTERRAD_H__
