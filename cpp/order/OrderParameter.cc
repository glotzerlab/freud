#include "OrderParameter.h"

namespace freud { namespace order {

HexOrderParameter::HexOrderParameter(unsigned int k)
    : OrderParameter<unsigned int>(k) {}

HexOrderParameter::~HexOrderParameter() {}

void HexOrderParameter::compute(const freud::locality::NeighborList* nlist,
                                const freud::locality::NeighborQuery* points,
                                freud::locality::QueryArgs qargs)
{
    computeGeneral(
    [this] (vec3<float> &delta)
    {
        float psi_ij = atan2f(delta.y, delta.x); 
        return exp(std::complex<float>(0, m_k * psi_ij));
    }, 
    nlist, points, qargs);
}

TransOrderParameter::TransOrderParameter(float k) 
    : OrderParameter<float>(k) {}

TransOrderParameter::~TransOrderParameter() {}

void TransOrderParameter::compute(const freud::locality::NeighborList* nlist,
                                  const freud::locality::NeighborQuery* points,
                                  freud::locality::QueryArgs qargs)
{
    computeGeneral(
    [] (vec3<float> &delta)
    {
        return std::complex<float>(delta.x, delta.y);
    }, 
    nlist, points, qargs);
}


}; };
