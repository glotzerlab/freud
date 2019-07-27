#ifndef NDHISTOGRAM_H
#define NDHISTOGRAM_H

#include <memory>

#include "Box.h"
#include "NeighborComputeFunctional.h"
#include "NeighborList.h"
#include "NeighborQuery.h"
#include "ThreadStorage.h"

namespace freud { namespace util {

template<typename T> std::shared_ptr<T> makeEmptyArray(unsigned int size)
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
    template<typename T> T reduceAndReturn(T thing_to_return)
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
    void accumulateGeneral(const locality::NeighborQuery* ref_points, const vec3<float>* points,
                           unsigned int n_p, const locality::NeighborList* nlist, unsigned int bin_size,
                           freud::locality::QueryArgs qargs, Func cf)
    {
        m_box = ref_points->getBox();
        locality::loopOverNeighbors(ref_points, points, n_p, qargs, nlist, cf);
        m_frame_counter++;
        m_n_ref = ref_points->getNRef();
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

    std::shared_ptr<float> m_pcf_array;         //!< Array of computed pair correlation function
    std::shared_ptr<unsigned int> m_bin_counts; //!< Counts for each bin
    util::ThreadStorage<unsigned int> m_local_bin_counts;
    //!< Thread local bin counts for TBB parallelism
};

}; }; // namespace freud::util

#endif
