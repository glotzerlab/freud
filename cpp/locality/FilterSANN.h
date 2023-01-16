#ifndef __FILTERSANN_H__
#define __FILTERSANN_H__

#include "Filter.h"

#include "NeighborList.h"
#include "NeighborQuery.h"

namespace freud { namespace locality {

/* Class for the SANN method of filtering a neighborlist.
 * */
class FilterSANN : public Filter
{
public:
    //<! Construct with an empty NeighborList, fill it upon calling compute
    FilterSANN() : Filter() {}

    void compute(const NeighborQuery* nq, const vec3<float>* query_points, unsigned int num_query_points,
                 const NeighborList* nlist, const QueryArgs& qargs) override;

private:
    void warnAboutUnfilledSolidAngles(const std::vector<unsigned int> &unfilled_qps);
};

}; }; // namespace freud::locality

#endif // __FILTERSANN_H__
