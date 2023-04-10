// Copyright (c) 2010-2023 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include "Steinhardt.h"
#include "NeighborComputeFunctional.h"
#include "utils.h"
#include <vector>

/*! \file Steinhardt.cc
    \brief Computes variants of Steinhardt order parameters.
*/

namespace freud { namespace order {

// Calculating Ylm using fsph module
void Steinhardt::computeYlm(fsph::PointSPHEvaluator<float>& sph_eval, const float theta, const float phi,
                            YlmsType& Ylms) const
{
    sph_eval.compute(theta, phi);

    for (size_t i = 0; i < m_ls.size(); ++i)
    {
        auto& Ylm = Ylms[i];
        const auto l = m_ls[i];
        size_t m_index(0);

        const auto end_iter(sph_eval.begin_l(l + 1, 0, true));
        for (auto iter(sph_eval.begin_l(l, 0, true)); iter != end_iter; ++iter, ++m_index)
        {
            // Manually add the Condon-Shortley phase, (-1)^m, to positive odd m
            const float phase = (m_index <= l && m_index % 2 == 1) ? -1 : 1;
            Ylm[m_index] = phase * (*iter);
        }
    }
}

void Steinhardt::reallocateArrays(unsigned int Np)
{
    m_Np = Np;

    const auto num_ls = m_ls.size();

    m_qli.prepare({Np, num_ls});
    if (m_average)
    {
        m_qliAve.prepare({Np, num_ls});
    }
    if (m_wl)
    {
        m_wli.prepare({Np, num_ls});
    }

    for (size_t l_index = 0; l_index < m_ls.size(); ++l_index)
    {
        const auto num_ms = m_num_ms[l_index];
        m_qlmi[l_index].prepare({Np, num_ms});
        m_qlm[l_index].prepare(num_ms);
        if (m_average)
        {
            m_qlmiAve[l_index].prepare({Np, num_ms});
        }
    }
}

void Steinhardt::compute(const freud::locality::NeighborList* nlist,
                         const freud::locality::NeighborQuery* points, freud::locality::QueryArgs qargs)
{
    // Allocate and zero out arrays as necessary.
    reallocateArrays(points->getNPoints());

    // Computes the base qlmi required for each specialized order parameter
    baseCompute(nlist, points, qargs);

    if (m_average)
    {
        computeAve(nlist, points, qargs);
    }

    // Reduce qlm
    for (size_t l_index = 0; l_index < m_ls.size(); ++l_index)
    {
        m_qlm_local[l_index].reduceInto(m_qlm[l_index]);
    }

    if (m_wl)
    {
        if (m_average)
        {
            aggregatewl(m_wli, m_qlmiAve, m_qliAve);
        }
        else
        {
            aggregatewl(m_wli, m_qlmi, m_qli);
        }
    }
    m_norm = normalizeSystem();
}

void Steinhardt::baseCompute(const freud::locality::NeighborList* nlist,
                             const freud::locality::NeighborQuery* points, freud::locality::QueryArgs qargs)
{
    std::vector<float> normalizationfactor(m_ls.size());
    for (size_t l_index = 0; l_index < m_ls.size(); ++l_index)
    {
        normalizationfactor[l_index] = float(4.0 * M_PI / m_num_ms[l_index]);
    }
    // For consistency, this reset is done here regardless of whether the array
    // is populated in baseCompute or computeAve.
    for (auto& qlm_local : m_qlm_local)
    {
        qlm_local.reset();
    }

    freud::locality::loopOverNeighborsIterator(
        points, points->getPoints(), m_Np, qargs, nlist,
        [&](size_t i, const std::shared_ptr<freud::locality::NeighborPerPointIterator>& ppiter) {
            float total_weight(0);
            const vec3<float> ref((*points)[i]);
            // Construct PointSPHEvaluator outside loop since the construction is costly.
            auto max_l = *std::max_element(m_ls.begin(), m_ls.end());
            fsph::PointSPHEvaluator<float> sph_eval(max_l);

            // Alocate and instantiate this array before looping over particles to prevent N instantiations
            // and N * m_l.size() allocations.
            YlmsType Ylms(m_ls.size());
            for (size_t l_index = 0; l_index < m_ls.size(); ++l_index)
            {
                Ylms[l_index].resize(m_num_ms[l_index]);
            }

            for (freud::locality::NeighborBond nb = ppiter->next(); !ppiter->end(); nb = ppiter->next())
            {
                const vec3<float> delta = points->getBox().wrap((*points)[nb.point_idx] - ref);
                const float weight(m_weighted ? nb.weight : float(1.0));

                // phi is usually in range 0..2Pi, but
                // it only appears in Ylm as exp(im\phi),
                // so range -Pi..Pi will give same results.
                const float phi = std::atan2(delta.y, delta.x); // -Pi..Pi

                // This value must be clamped in cases where the particles are
                // aligned along z, otherwise due to floating point error we
                // could get delta.z/nb.distance = -1-eps, which is outside the
                // valid range of std::acos.
                float theta = std::acos(util::clamp(delta.z / nb.distance, -1, 1)); // 0..Pi

                // If the points are directly on top of each other,
                // theta should be zero instead of nan.
                if (nb.distance == float(0))
                {
                    theta = 0;
                }

                computeYlm(sph_eval, theta, phi, Ylms); // Fill up Ylm

                for (size_t l_index = 0; l_index < m_ls.size(); ++l_index)
                {
                    auto& qlmi = m_qlmi[l_index];
                    const auto& Ylm = Ylms[l_index];
                    // Get the initial index and iterate using ++ for faster iteration
                    // Profiling showed using operator() to slow the code significantly.
                    const size_t index = qlmi.getIndex({i, 0});
                    for (size_t k = 0; k < m_num_ms[l_index]; ++k)
                    {
                        qlmi[index + k] += weight * Ylm[k];
                    }
                }
                // Accumulate weight for normalization
                total_weight += weight;
            } // End loop going over neighbor bonds

            // Normalize!
            const size_t qli_i_start = m_qli.getIndex({i, 0});
            for (size_t l_index = 0; l_index < m_ls.size(); ++l_index)
            {
                // get l specific vectors/arrays
                auto& qlmi = m_qlmi[l_index];
                auto& qlm_local = m_qlm_local[l_index];
                const size_t first_qlmi_index = qlmi.getIndex({i, 0});
                const size_t qli_index = qli_i_start + l_index;

                for (size_t k = 0; k < m_num_ms[l_index]; ++k)
                {
                    // Cache the index for efficiency.
                    const size_t qlmi_index = first_qlmi_index + k;

                    qlmi[qlmi_index] /= total_weight;
                    // Add the norm, which is the (complex) squared magnitude
                    m_qli[qli_index] += norm(qlmi[qlmi_index]);
                    // This array gets populated by computeAve in the averaging case.
                    if (!m_average)
                    {
                        qlm_local.local()[k] += qlmi[qlmi_index] / float(m_Np);
                    }
                }
                m_qli[qli_index] *= normalizationfactor[l_index];
                m_qli[qli_index] = std::sqrt(m_qli[qli_index]);
            }
        });
}

void Steinhardt::computeAve(const freud::locality::NeighborList* nlist,
                            const freud::locality::NeighborQuery* points, freud::locality::QueryArgs qargs)
{
    std::shared_ptr<locality::NeighborQueryIterator> iter;
    if (nlist == nullptr)
    {
        iter = points->query(points->getPoints(), points->getNPoints(), qargs);
    }

    std::vector<float> normalizationfactor(m_ls.size());
    for (size_t l_index = 0; l_index < m_ls.size(); ++l_index)
    {
        normalizationfactor[l_index] = static_cast<float>(4.0 * M_PI / m_num_ms[l_index]);
    }

    freud::locality::loopOverNeighborsIterator(
        points, points->getPoints(), m_Np, qargs, nlist,
        [&](size_t i, const std::shared_ptr<freud::locality::NeighborPerPointIterator>& ppiter) {
            unsigned int neighborcount(1);
            for (freud::locality::NeighborBond nb = ppiter->next(); !ppiter->end(); nb = ppiter->next())
            {
                for (size_t l_index = 0; l_index < m_ls.size(); ++l_index)
                {
                    auto& qlmiAve = m_qlmiAve[l_index];
                    auto& qlmi = m_qlmi[l_index];
                    const auto ave_index = qlmiAve.getIndex({i, 0});
                    const auto nb_index = qlmi.getIndex({nb.point_idx, 0});
                    for (size_t k = 0; k < m_num_ms[l_index]; ++k)
                    {
                        // Adding all the qlm of the neighbors. We use the
                        // vector function signature for indexing into the
                        // arrays for speed.
                        qlmiAve[ave_index + k] += qlmi[nb_index + k];
                    }
                }
                neighborcount++;
            } // End loop over particle's bonds

            // Normalize!

            const size_t qliAve_i_start = m_qliAve.getIndex({i, 0});
            for (size_t l_index = 0; l_index < m_ls.size(); ++l_index)
            {
                auto& qlmiAve = m_qlmiAve[l_index];
                auto& qlmi = m_qlmi[l_index];
                auto& qlm_local = m_qlm_local[l_index];
                const size_t first_qlmi_index = qlmiAve.getIndex({i, 0});
                const size_t qliAve_index = qliAve_i_start + l_index;

                for (size_t k = 0; k < m_num_ms[l_index]; ++k)
                {
                    // Cache the index for efficiency.
                    const size_t qlmi_index = first_qlmi_index + k;
                    // Add the qlm of the particle i itself
                    qlmiAve[qlmi_index] += qlmi[qlmi_index];
                    qlmiAve[qlmi_index] /= static_cast<float>(neighborcount);
                    qlm_local.local()[k] += qlmiAve[qlmi_index] / float(m_Np);
                    // Add the norm, which is the complex squared magnitude
                    m_qliAve[qliAve_index] += norm(qlmiAve[qlmi_index]);
                }
                m_qliAve[qliAve_index] *= normalizationfactor[l_index];
                m_qliAve[qliAve_index] = std::sqrt(m_qliAve[qliAve_index]);
            }
        });
}

std::vector<float> Steinhardt::normalizeSystem()
{
    std::vector<float> system_norms(m_ls.size());
    for (size_t l_index = 0; l_index < m_ls.size(); ++l_index)
    {
        auto& qlm = m_qlm[l_index];
        auto l = m_ls[l_index];
        float calc_norm(0);
        const auto normalizationfactor = static_cast<float>(4.0 * M_PI / m_num_ms[l_index]);
        for (size_t k = 0; k < m_num_ms[l_index]; ++k)
        {
            // Add the norm, which is the complex squared magnitude
            calc_norm += norm(qlm[k]);
        }
        const float ql_system_norm = std::sqrt(calc_norm * normalizationfactor);

        if (m_wl)
        {
            const auto wigner3j_values = getWigner3j(l);
            float wl_system_norm = reduceWigner3j(qlm.get(), l, wigner3j_values);

            // The normalization factor of wl is calculated using qli, which is
            // equivalent to calculate the normalization factor from qlmi
            if (m_wl_normalize)
            {
                const float wl_normalization = std::sqrt(normalizationfactor) / ql_system_norm;
                wl_system_norm *= wl_normalization * wl_normalization * wl_normalization;
            }
            system_norms[l_index] = wl_system_norm;
        }
        else
        {
            system_norms[l_index] = ql_system_norm;
        }
    }
    return system_norms;
}

void Steinhardt::aggregatewl(util::ManagedArray<float>& target,
                             const std::vector<util::ManagedArray<std::complex<float>>>& source,
                             const util::ManagedArray<float>& normalization_source) const
{
    util::forLoopWrapper(0, m_Np, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i)
        {
            const auto target_particle_index = target.getIndex({i, 0});
            const auto norm_particle_index = normalization_source.getIndex({i, 0});
            for (size_t l_index = 0; l_index < m_ls.size(); ++l_index)
            {
                const auto l = m_ls[l_index];
                const auto& source_l = source[l_index];

                const auto normalizationfactor = static_cast<float>(4.0 * M_PI / m_num_ms[l_index]);
                const auto wigner3j_values = getWigner3j(l);

                target[target_particle_index + l_index]
                    = reduceWigner3j(&(source_l({i, 0})), l, wigner3j_values);
                if (m_wl_normalize)
                {
                    const float normalization = std::sqrt(normalizationfactor)
                        / normalization_source[norm_particle_index + l_index];
                    target[target_particle_index + l_index] *= normalization * normalization * normalization;
                }
            }
        }
    });
}

}; }; // end namespace freud::order
