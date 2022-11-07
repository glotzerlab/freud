#include "NeighborList.h"
#include "NeighborQuery.h"

namespace freud { namespace locality {

class Filter
{
public:
    Filter() : m_unfiltereed_nlist(std::make_shared<NeighborList>()), m_filtered_nlist(std::make_shared<NeighborList>()) {}

    void compute(const NeighborQuery *nq, const vec3<float> *query_points,
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

private:
    //!< The unfiltered neighborlist
    std::shared_ptr<NeighborList> m_unfiltered_nlist;
    //!< The filtered neighborlist
    std::shared_ptr<NeighborList> m_filtered_nlist;
}

}; };
