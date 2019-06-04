// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef PMFT_H
#define PMFT_H

#include <memory>
#include <ostream>
#include <tbb/tbb.h>

#include "Box.h"
#include "Index1D.h"
#include "NeighborList.h"
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
class PMFT
{
public:
    //! Constructor
    PMFT();

    //! Destructor
    virtual ~PMFT();

    //! Get the simulation box
    const box::Box& getBox() const
    {
        return m_box;
    }

    //! Reset the PCF array to all zeros
    virtual void reset() = 0;

    //! \internal
    //! helper function to reduce the thread specific arrays into one array
    //! Must be implemented by subclasses
    virtual void reducePCF() = 0;

    //! Get a reference to the PCF array
    std::shared_ptr<float> getPCF();

    //! Get a reference to the bin counts array
    std::shared_ptr<unsigned int> getBinCounts();

    float getRCut()
    {
        return m_r_cut;
    }

    std::shared_ptr<float> precomputeAxisBinCenter(unsigned int size, float d, float max);


template<typename Func> std::shared_ptr<float> precomputeArrayGeneral(unsigned int size, float d, Func cf)
{
    std::shared_ptr<float> arr = std::shared_ptr<float>(new float[size], std::default_delete<float[]>());
    for (unsigned int i = 0; i < size; i++)
    {
        float T = float(i) * d;
        float nextT = float(i + 1) * d;
        arr.get()[i] = cf(T, nextT);
    }
    return arr;
}

template<typename Func>
void accumulateGeneral(box::Box& box, unsigned int n_ref, const locality::NeighborList* nlist,
                       unsigned int n_p, unsigned int bin_size, Func cf)
{
    m_box = box;
    nlist->validate(n_ref, n_p);
    const size_t* neighbor_list(nlist->getNeighbors());

    tbb::parallel_for(tbb::blocked_range<size_t>(0, n_ref), [=](const tbb::blocked_range<size_t>& r) {
        bool exists;
        m_local_bin_counts.local(exists);
        if (!exists)
        {
            m_local_bin_counts.local() = new unsigned int[bin_size];
            memset((void*) m_local_bin_counts.local(), 0, sizeof(unsigned int) * bin_size);
        }
        size_t bond(nlist->find_first_index(r.begin()));
        // for each reference point
        for (size_t i = r.begin(); i != r.end(); i++)
        {
            for (; bond < nlist->getNumBonds() && neighbor_list[2 * bond] == i; ++bond)
            {
                const size_t j(neighbor_list[2 * bond + 1]);
                cf(i, j);
            }
        } // done looping over reference points
    });
    this->m_frame_counter++;
    this->m_n_ref = n_ref;
    this->m_n_p = n_p;
    // flag to reduce
    this->m_reduce = true;
}

    void resetGeneral(unsigned int bin_size);


template<typename JacobFactor>
void reduce2D(unsigned int first_dim, unsigned int second_dim, JacobFactor jf)
{
    reduce3D(1, first_dim, second_dim, jf);
}

template<typename JacobFactor>
void reduce3D(unsigned int n_r, unsigned int first_dim, unsigned int second_dim, JacobFactor jf)
{
    memset((void*) m_bin_counts.get(), 0, sizeof(unsigned int) * n_r * first_dim * second_dim);
    memset((void*) m_pcf_array.get(), 0, sizeof(float) * n_r * first_dim * second_dim);
    parallel_for(tbb::blocked_range<size_t>(0, first_dim), [=](const tbb::blocked_range<size_t>& r) {
        Index3D b_i = Index3D(first_dim, second_dim, n_r);
        for (size_t i = r.begin(); i != r.end(); i++)
        {
            for (size_t j = 0; j < second_dim; j++)
            {
                for (size_t k = 0; k < n_r; k++)
                {
                    for (tbb::enumerable_thread_specific<unsigned int*>::const_iterator local_bins
                         = m_local_bin_counts.begin();
                         local_bins != m_local_bin_counts.end(); ++local_bins)
                    {
                        m_bin_counts.get()[b_i((int) i, (int) j, (int) k)]
                            += (*local_bins)[b_i((int) i, (int) j, (int) k)];
                    }
                }
            }
        }
    });
    float inv_num_dens = m_box.getVolume() / (float) this->m_n_p;
    float norm_factor = (float) 1.0 / ((float) this->m_frame_counter * (float) this->m_n_ref);
    // normalize pcf_array
    // avoid need to unravel b/c arrays are in the same index order
    parallel_for(tbb::blocked_range<size_t>(0, n_r * first_dim * second_dim),
                 [=](const tbb::blocked_range<size_t>& r) {
                     for (size_t i = r.begin(); i != r.end(); i++)
                     {
                         m_pcf_array.get()[i]
                             = (float) m_bin_counts.get()[i] * norm_factor * jf(i) * inv_num_dens;
                     }
                 });
}
protected:
    box::Box m_box;               //!< Simulation box where the particles belong
    float m_r_cut;                //!< r_cut used in cell list construction
    unsigned int m_frame_counter; //!< Number of frames calculated
    unsigned int m_n_ref;         //!< The number of reference points
    unsigned int m_n_p;           //!< The number of points
    bool m_reduce;                //!< Whether or not the PCF needs to be reduced

    std::shared_ptr<float> m_pcf_array;         //!< Array of PCF computed
    std::shared_ptr<unsigned int> m_bin_counts; //!< Counts for each bin
    tbb::enumerable_thread_specific<unsigned int*>
        m_local_bin_counts; //!< Thread local bin counts for TBB parallelism

private:
};

}; }; // end namespace freud::pmft

#endif // PMFT_H
