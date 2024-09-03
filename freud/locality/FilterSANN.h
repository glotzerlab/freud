// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#pragma once

#include "Filter.h"

#include "NeighborList.h"
#include "NeighborQuery.h"
#include <memory>
#include "VectorMath.h"
#include <vector>

namespace freud { namespace locality {

/* Class for the SANN method of filtering a neighborlist.
 * */
class FilterSANN : public Filter
{
public:
    //<! Construct a FilterSANN
    /*
     * \param allow_incomplete_shell whether incomplete neighbor shells should
     *                               result in a warning or error
     * */
    explicit FilterSANN(bool allow_incomplete_shell) : Filter(allow_incomplete_shell) {}

    void compute(std::shared_ptr<NeighborQuery> nq, const vec3<float>* query_points,
                 unsigned int num_query_points, std::shared_ptr<NeighborList> nlist,
                 const QueryArgs& qargs) override;

private:
    //! warn/raise exception about unfilled shells
    void warnAboutUnfilledNeighborShells(const std::vector<unsigned int>& unfilled_qps) const;
};

}; }; // namespace freud::locality
