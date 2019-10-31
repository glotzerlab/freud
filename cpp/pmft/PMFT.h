// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef PMFT_H
#define PMFT_H

#include <tbb/tbb.h>

#include "BondHistogramCompute.h"
#include "Box.h"
#include "Histogram.h"
#include "ManagedArray.h"
#include "VectorMath.h"

/*! \internal
    \file PMFT.h
    \brief Declares base class for all PMFT classes
*/

namespace freud { namespace pmft {

//! Computes the PMFT for a given set of points
/*! The PMFT class is an abstract class providing the basis for all classes calculating PMFTs for specific
 *  dimensional cases. The PMFT class defines some of the key interfaces required for all PMFT classes, such
 *  as the ability to access the underlying PCF and box. Many of the specific methods must be implemented by
 *  subclasses that account for the proper set of dimensions.The required functions are implemented as pure
 *  virtual functions here to enforce this.
 */
class PMFT : public locality::BondHistogramCompute
{
public:
    //! Constructor
    PMFT() : BondHistogramCompute() {}

    //! Destructor
    virtual ~PMFT() {};

    //! \internal
    // Wrapper to do accumulation.
    /*! \param neighbor_query NeighborQuery object to iterate over
        \param query_points Points
        \param n_query_points Number of query_points
        \param nlist Neighbor List. If not NULL, loop over it. Otherwise, use neighbor_query
           appropriately with given qargs.
        \param qargs Query arguments
        \param cf An object with operator(NeighborBond) as input.
    */
    template<typename Func>
    void accumulateGeneral(const locality::NeighborQuery* neighbor_query, const vec3<float>* query_points,
                           unsigned int n_query_points, const locality::NeighborList* nlist,
                           freud::locality::QueryArgs qargs, Func cf)
    {
        m_box = neighbor_query->getBox();
        locality::loopOverNeighbors(neighbor_query, query_points, n_query_points, qargs, nlist, cf);
        m_frame_counter++;
        m_n_points = neighbor_query->getNPoints();
        m_n_query_points = n_query_points;
        // flag to reduce
        m_reduce = true;
    }

    template<typename JacobFactor> void reduce(JacobFactor jf)
    {
        m_pcf_array.prepare(m_histogram.shape());
        m_histogram.prepare(m_histogram.shape());

        float inv_num_dens = m_box.getVolume() / (float) m_n_query_points;
        float norm_factor = (float) 1.0 / ((float) m_frame_counter * (float) m_n_points);
        float prefactor = inv_num_dens * norm_factor;

        m_histogram.reduceOverThreadsPerBin(m_local_histograms, [this, &prefactor, &jf](size_t i) {
            m_pcf_array[i] = m_histogram[i] * prefactor * jf(i);
        });
    }

    //! Get a reference to the PCF array
    const util::ManagedArray<float>& getPCF()
    {
        return reduceAndReturn(m_pcf_array);
    }

protected:
    util::ManagedArray<float> m_pcf_array; //!< Array of computed pair correlation function.
};

}; }; // end namespace freud::pmft

#endif // PMFT_H
