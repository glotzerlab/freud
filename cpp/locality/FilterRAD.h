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
    FilterRAD() : Filter() {}

    void compute(const NeighborQuery* nq, const vec3<float>* query_points, unsigned int num_query_points,
                 const NeighborList* nlist, const QueryArgs& qargs) override;
};

}; }; // namespace freud::locality

#endif // __FILTERRAD_H__
