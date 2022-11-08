#ifndef __FILTER_H__
#define __FILTER_H__

#include "NeighborList.h"
#include "NeighborQuery.h"

namespace freud { namespace locality {

class Filter
{
public:
    Filter() : m_unfiltered_nlist(std::make_shared<NeighborList>()), m_filtered_nlist(std::make_shared<NeighborList>()) {}

    virtual ~Filter() {}

    virtual void compute(const NeighborQuery *nq, const vec3<float> *query_points,
            unsigned int num_query_points,
            const NeighborList *nlist, QueryArgs qargs) = 0;

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
};

}; };

#endif // __FILTER_H__
