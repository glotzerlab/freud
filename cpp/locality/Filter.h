// Copyright (c) 2010-2023 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef __FILTER_H__
#define __FILTER_H__

#include "NeighborList.h"
#include "NeighborQuery.h"
#include <iostream>

namespace freud { namespace locality {

/* Base class for all Neigborlist filtering methods in freud.
 *
 * A neighborlist filter is a class which is given a neighborlist and its goal
 * is to return a neighborlist with a number of bonds less than or equal to the
 * given neighborlist. Each filter will use information about the system to
 * determine which bonds need to be removed.
 *
 * After calling compute(), both the original Neighborlist and the new, filtered
 * neighborlist will be made available to users on the python side.
 * */
class Filter
{
public:
    Filter(bool allow_incomplete_shell)
        : m_unfiltered_nlist(std::make_shared<NeighborList>()),
          m_filtered_nlist(std::make_shared<NeighborList>()), m_allow_incomplete_shell(allow_incomplete_shell)
    {}

    virtual ~Filter() = default;

    virtual void compute(const NeighborQuery* nq, const vec3<float>* query_points,
                         unsigned int num_query_points, const NeighborList* nlist, const QueryArgs& qargs)
        = 0;

    std::shared_ptr<NeighborList> getFilteredNlist() const
    {
        return m_filtered_nlist;
    }

    std::shared_ptr<NeighborList> getUnfilteredNlist() const
    {
        return m_unfiltered_nlist;
    }

protected:
    //!< The unfiltered neighborlist
    std::shared_ptr<NeighborList> m_unfiltered_nlist;

    //!< The filtered neighborlist
    std::shared_ptr<NeighborList> m_filtered_nlist;

    //<! whether a warning (true) or error (false) should be raised if the filter
    //<! algorithm implementation cannot guarantee that all neighbors have completely filled shells
    bool m_allow_incomplete_shell;

    /*! Output the appropriate warning/error message for particles with unfilled shells
     *
     * In general, the filter concept cannot guarantee that each query point will have
     * a completely filled shell according to the implemented algorithm. This happens
     * when the initial unfiltered neighbor list doesn't have enough neighbors to
     * guarantee this criterion.
     *
     * \param unfilled_qps Vector of query points which may have unfilled neighbor shells.
     *                     The vector should have the value
     *                     ``unfilled_qps[i] = std::numeric_limits<unsigned int>::max()``
     *                     for all query point indices ``i`` which have filled shells, and
     *                     should have ``unfilled_qps[i] = i`` for all query point indices
     *                     ``i`` which may not have a filled neighbor shell.
     * */
    void warnAboutUnfilledNeighborShells(const std::vector<unsigned int>& unfilled_qps) const
    {
        std::string indices;
        for (const auto& idx : unfilled_qps)
        {
            if (idx != std::numeric_limits<unsigned int>::max())
            {
                indices += std::to_string(idx);
                indices += ", ";
            }
        }
        indices = indices.substr(0, indices.size() - 2);
        if (!indices.empty())
        {
            std::ostringstream error_str;
            error_str << "Query point indices " << indices << " do not have full neighbor shells.";
            if (!m_allow_incomplete_shell)
            {
                throw std::runtime_error(error_str.str());
            }
            std::cout << "WARNING: " << error_str.str() << std::endl;
        }
    }
};

}; }; // namespace freud::locality

#endif // __FILTER_H__
