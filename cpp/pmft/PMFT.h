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

template<typename T>
std::shared_ptr<T> returnEmptyArray(unsigned int size)
{
    auto new_arr = std::shared_ptr<T>(new T[size], std::default_delete<T[]>());
    memset((void*) new_arr.get(), 0, sizeof(T) * size);
    return new_arr;
}

//! Parent class for PMFT and RDF
class NdHistogram
{
public:
    //! Constructor
    NdHistogram();

    //! Destructor
    virtual ~NdHistogram() {};

    //! \internal
    //! Pure virtual reduce function to make :code:`reduceAndReturn` work.
    virtual void reduce() = 0;

    //! Return :code:`thing_to_return` after reducing.
    template<typename T>
    T reduceAndReturn(T thing_to_return)
    {
        if (m_reduce == true)
        {
            reduce();
        }
        m_reduce = false;
        return thing_to_return;
    }

    //! Get a reference to the PCF array
    std::shared_ptr<float> getPCF() 
    {
        return reduceAndReturn(m_pcf_array);
    }

    //! Get a reference to the bin counts array
    std::shared_ptr<unsigned int> getBinCounts()
    {
        return reduceAndReturn(m_bin_counts);
    }

    //! Get the simulation box
    const box::Box& getBox() const
    {
        return m_box;
    }

    //! \internal
    //! Reset m_local_bin_counts
    void resetGeneral(unsigned int bin_size);

    //! \internal
    // Wrapper to do accumulation.
    // :code:`Func cf` should be some sort of (void*)(size_t, size_t)
    template<typename Func>
    void accumulateGeneral(box::Box& box, unsigned int n_ref, const locality::NeighborList* nlist,
                           unsigned int n_p, unsigned int bin_size, Func cf)
    {
        m_box = box;
        nlist->validate(n_ref, n_p);
        const size_t* neighbor_list(nlist->getNeighbors());
        size_t n_bonds = nlist->getNumBonds();
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n_bonds),
            [=] (const tbb::blocked_range<size_t>& r)
            {
                 bool exists;
                m_local_bin_counts.local(exists);
                if (!exists)
                {
                    m_local_bin_counts.local() = new unsigned int[bin_size];
                    memset((void*) m_local_bin_counts.local(), 0, sizeof(unsigned int) * bin_size);
                }
                for(size_t bond = r.begin(); bond !=r.end(); ++bond)
                    {
                    size_t i(neighbor_list[2*bond]);
                    size_t j(neighbor_list[2*bond + 1]);
                    cf(i, j);
                    }
            });
        m_frame_counter++;
        m_n_ref = n_ref;
        m_n_p = n_p;
        // flag to reduce
        m_reduce = true;
    }

    protected:
        box::Box m_box;
        unsigned int m_frame_counter; //!< Number of frames calculated
        unsigned int m_n_ref;         //!< The number of reference points
        unsigned int m_n_p;           //!< The number of points
        bool m_reduce;                //!< Whether or not the PCF needs to be reduced

        std::shared_ptr<float> m_pcf_array;         //!< Array of PCF/RDF computed 
        std::shared_ptr<unsigned int> m_bin_counts; //!< Counts for each bin
        tbb::enumerable_thread_specific<unsigned int*>
            m_local_bin_counts; //!< Thread local bin counts for TBB parallelism

};

//! Computes the PMFT for a given set of points
/*! The PMFT class is an abstract class providing the basis for all classes calculating PMFTs for specific
 *  dimensional cases. The PMFT class defines some of the key interfaces required for all PMFT classes, such
 *  as the ability to access the underlying PCF and box. Many of the specific methods must be implemented by
 *  subclasses that account for the proper set of dimensions.The required functions are implemented as pure
 *  virtual functions here to enforce this.
 */
class PMFT : public NdHistogram
{
public:
    //! Constructor
    PMFT();

    //! Destructor
    virtual ~PMFT();

    //! Reset the PCF array to all zeros
    virtual void reset() = 0;

    //! \internal
    //! helper function to reduce the thread specific arrays into one array
    //! Must be implemented by subclasses
    virtual void reducePCF() = 0;

    //! Implementing pure virtual function from parent class.
    virtual void reduce()
    {
        reducePCF();
    }

    float getRCut()
    {
        return m_r_cut;
    }

    //! Helper function to precompute axis bin center,
    std::shared_ptr<float> precomputeAxisBinCenter(unsigned int size, float d, float max);

    //! Helper function to precompute array with the following logic.
    //! :cpde:`Func cf` should be some sort of (float)(float, float).
    template<typename Func> 
    std::shared_ptr<float> precomputeArrayGeneral(unsigned int size, float d, Func cf)
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

    //! Helper function to reduce two dimensionally with appropriate Jaocobian.
    template<typename JacobFactor>
    void reduce2D(unsigned int first_dim, unsigned int second_dim, JacobFactor jf)
    {
        reduce3D(1, first_dim, second_dim, jf);
    }

    //! Helper function to reduce three dimensionally with appropriate Jaocobian.
    template<typename JacobFactor>
    void reduce3D(unsigned int n_r, unsigned int first_dim, unsigned int second_dim, JacobFactor jf)
    {
        unsigned int loocal_bin_counts_size = n_r * first_dim * second_dim;
        memset((void*) m_bin_counts.get(), 0, sizeof(unsigned int) * loocal_bin_counts_size);
        memset((void*) m_pcf_array.get(), 0, sizeof(float) * loocal_bin_counts_size);
        parallel_for(tbb::blocked_range<size_t>(0, loocal_bin_counts_size), [=](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); i++)
            {
                for (tbb::enumerable_thread_specific<unsigned int*>::const_iterator local_bins
                     = m_local_bin_counts.begin();
                     local_bins != m_local_bin_counts.end(); ++local_bins)
                {
                    m_bin_counts.get()[i] += (*local_bins)[i];
                }
            }
        });
        float inv_num_dens = m_box.getVolume() / (float) m_n_p;
        float norm_factor = (float) 1.0 / ((float) m_frame_counter * (float) m_n_ref);
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
    float m_r_cut;                //!< r_cut used in cell list construction
private:
};

}; }; // end namespace freud::pmft

#endif // PMFT_H
