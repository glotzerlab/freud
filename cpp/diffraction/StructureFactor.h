// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef STRUCTURE_FACTOR_H
#define STRUCTURE_FACTOR_H

#include "Box.h"
#include "Histogram.h"
#include "NeighborQuery.h"

/*! \file StructureFactor.h
    \brief Routines for computing static structure factors.

    Computes structure factors from the Fourier transform of a radial distribution function (RDF).
    This method is not capable of resolving k-vectors smaller than the magnitude 4 * pi / L, where L is the
    smallest side length of the system's periodic box.
*/

namespace freud { namespace diffraction {

class StructureFactor
{
    using StructureFactorHistogram = util::Histogram<float>;

public:
    //! Constructor
    StructureFactor(unsigned int bins, float k_max, float k_min = 0);

    //! Destructor
    virtual ~StructureFactor() {};

    //! Compute the structure factor
    /*! Computes an RDF and integrates the RDF to get the structure factor S(k).
     *  Both steps are parallelized.
     */
    void accumulate(const freud::locality::NeighborQuery* neighbor_query, const vec3<float>* query_points,
                    unsigned int n_query_points, const freud::locality::NeighborList* nlist,
                    freud::locality::QueryArgs qargs);

    //! Get the structure factor
    const util::ManagedArray<float>& getStructureFactor();

    //! Get the k bin edges
    const std::vector<float> getBinEdges() const
    {
        return m_histogram.getBinEdges()[0];
    }

    //! Get the k bin centers
    const std::vector<float> getBinCenters() const
    {
        return m_histogram.getBinCenters()[0];
    }

    //! Get the minimum valid k value
    float getMinValidK() const
    {
        return m_min_valid_k;
    }

private:
    StructureFactorHistogram m_histogram; //!< Histogram to hold computed structure factor
    StructureFactorHistogram::ThreadLocalHistogram
        m_local_histograms;                       //!< Thread local histograms for TBB parallelism
    util::ManagedArray<float> m_structure_factor; //!< The computed structure factor
    float m_min_valid_k;                          //!< The minimum valid k-vector based on the computed box
    bool m_reduce;                                //!< Whether to reduce
};

}; }; // namespace freud::diffraction

#endif // STRUCTURE_FACTOR_H
