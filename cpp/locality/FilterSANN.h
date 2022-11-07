#include "Filter.h"

#include "NeighborList.h"
#include "NeighborQuery.h"

namespace freud { namespace locality {

class FilterSANN : public Filter
{
public:
    //<! Construct with an empty NeighborList, fill it upon calling compute
    FilterSANN() : Filter() {}

    void compute(const NeigbborQuery *nq, const vec3<float> *query_points,
                    unsigned int num_query_points, const NeighborList *nlist,
                    QueryArgs qargs) override;
}

}; };
