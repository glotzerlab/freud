// Copyright (c) 2010-2025 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <vector>

#include "HexaticTranslational.h"
#include "ManagedArray.h"
#include "NeighborBond.h"
#include "NeighborComputeFunctional.h"
#include "NeighborList.h"
#include "NeighborPerPointIterator.h"
#include "NeighborQuery.h"
#include "VectorMath.h"

namespace freud { namespace order {

//! Compute the order parameter
template<typename T>
template<typename Func>
void HexaticTranslational<T>::computeGeneral(Func func, const std::shared_ptr<locality::NeighborList>& nlist,
                                             const std::shared_ptr<locality::NeighborQuery>& points,
                                             const freud::locality::QueryArgs qargs, bool normalize_by_k)
{
    const auto box = points->getBox();
    box.enforce2D();

    const unsigned int Np = points->getNPoints();

    m_psi_array = std::make_shared<util::ManagedArray<std::complex<float>>>(std::vector<size_t> {Np});

    freud::locality::loopOverNeighborsIterator(
        points, points->getPoints(), Np, qargs, nlist,
        [&](size_t i, const std::shared_ptr<freud::locality::NeighborPerPointIterator>& ppiter) {
            float total_weight(0);
            const vec3<float> ref((*points)[i]);

            for (freud::locality::NeighborBond nb = ppiter->next(); !ppiter->end(); nb = ppiter->next())
            {
                // Compute vector from query_point to point
                const vec3<float> delta = box.wrap((*points)[nb.getPointIdx()] - ref);
                const float weight(m_weighted ? nb.getWeight() : 1.0);

                // Compute psi for this vector
                (*m_psi_array)[i] += weight * func(delta);
                total_weight += weight;
            }
            if (normalize_by_k)
            {
                (*m_psi_array)[i] /= std::complex<float>(m_k);
            }
            else
            {
                (*m_psi_array)[i] /= std::complex<float>(total_weight);
            }
        });
}

Hexatic::Hexatic(unsigned int k, bool weighted) : HexaticTranslational<unsigned int>(k, weighted) {}

void Hexatic::compute(const std::shared_ptr<locality::NeighborList>& nlist,
                      const std::shared_ptr<locality::NeighborQuery>& points,
                      const freud::locality::QueryArgs& qargs)
{
    computeGeneral(
        [this](const vec3<float>& delta) {
            const float theta_ij = std::atan2(delta.y, delta.x);
            return std::exp(std::complex<float>(0, static_cast<float>(m_k) * theta_ij));
        },
        nlist, points, qargs, false);
}

}; }; // namespace freud::order
