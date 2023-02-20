#ifndef __FILTER_H__
#define __FILTER_H__

#include <iostream>
#include "NeighborList.h"
#include "NeighborQuery.h"

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
          m_filtered_nlist(std::make_shared<NeighborList>()),
          m_allow_incomplete_shell(allow_incomplete_shell)
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
    //<! variable that determines if RAD open (True) or RAD closed (False) is computed
    bool m_allow_incomplete_shell;

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
