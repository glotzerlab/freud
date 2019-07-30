#include "HexTransOrderParameter.h"

namespace freud { namespace order {

//! Compute the order parameter
template<typename T>
template<typename Func>
void HexTransOrderParameter<T>::computeGeneral(Func func, const freud::locality::NeighborList* nlist,
                              const freud::locality::NeighborQuery* points, freud::locality::QueryArgs qargs)
{
    // Compute the cell list
    m_box = points->getBox();
    unsigned int Np = points->getNPoints();

    // Reallocate the output array if it is not the right size
    if (Np != m_Np)
    {
        m_psi_array = std::shared_ptr<std::complex<float>>(new std::complex<float>[Np],
                                                      std::default_delete<std::complex<float>[]>());
    }

    freud::locality::loopOverNeighborsIterator(points, points->getPoints(), Np, qargs, nlist, 
    [=] (size_t i, std::shared_ptr<freud::locality::NeighborIterator::PerPointIterator> ppiter)
    {
        m_psi_array.get()[i] = 0;
        vec3<float> ref = (*points)[i];

        for(freud::locality::NeighborBond nb = ppiter->next(); !ppiter->end(); nb = ppiter->next())
        {
            // Compute r between the two particles
            vec3<float> delta = m_box.wrap((*points)[nb.ref_id] - ref);

            // Compute psi for neighboring particle
            // (only constructed for 2d)
            m_psi_array.get()[i] += func(delta);
        }

        m_psi_array.get()[i] /= std::complex<float>(m_k);
    });
    // Save the last computed number of particles
    m_Np = Np;
}

HexOrderParameter::HexOrderParameter(unsigned int k)
    : HexTransOrderParameter<unsigned int>(k) {}

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
    : HexTransOrderParameter<float>(k) {}

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
