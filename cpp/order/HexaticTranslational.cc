// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include "HexaticTranslational.h"

namespace freud { namespace order {

//! Compute the order parameter
template<typename T>
template<typename Func>
void HexaticTranslational<T>::computeGeneral(Func func, const freud::locality::NeighborList* nlist,
                                             const freud::locality::NeighborQuery* points,
                                             freud::locality::QueryArgs qargs)
{
    const auto box = points->getBox();
    box.enforce2D();

    const unsigned int Np = points->getNPoints();

    m_psi_array.prepare(Np);

    freud::locality::loopOverNeighborsIterator(
        points, points->getPoints(), Np, qargs, nlist,
        [=](size_t i, std::shared_ptr<freud::locality::NeighborPerPointIterator> ppiter) {
            const vec3<float> ref = (*points)[i];

            for (freud::locality::NeighborBond nb = ppiter->next(); !ppiter->end(); nb = ppiter->next())
            {
                // Compute vector from query_point to point
                const vec3<float> delta = box.wrap((*points)[nb.point_idx] - ref);

                // Compute psi for this vector
                m_psi_array[i] += func(delta);
            }

            m_psi_array[i] /= std::complex<float>(m_k);
        });
}

Hexatic::Hexatic(unsigned int k) : HexaticTranslational<unsigned int>(k) {}

Hexatic::~Hexatic() {}

void Hexatic::compute(const freud::locality::NeighborList* nlist,
                      const freud::locality::NeighborQuery* points, freud::locality::QueryArgs qargs)
{
    computeGeneral(
        [this](const vec3<float>& delta) {
            const float psi_ij = atan2f(delta.y, delta.x);
            return exp(std::complex<float>(0, m_k * psi_ij));
        },
        nlist, points, qargs);
}

Translational::Translational(float k) : HexaticTranslational<float>(k) {}

Translational::~Translational() {}

void Translational::compute(const freud::locality::NeighborList* nlist,
                            const freud::locality::NeighborQuery* points, freud::locality::QueryArgs qargs)
{
    computeGeneral([](const vec3<float>& delta) { return std::complex<float>(delta.x, delta.y); }, nlist,
                   points, qargs);
}

}; }; // namespace freud::order
