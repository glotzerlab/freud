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
    /*! \param neighbor_query NeighborQuery object to iterate over
        \param query_points Points
        \param n_query_points Number of query_points
        \param nlist Neighbor List. If not NULL, loop over it. Otherwise, use neighbor_query
           appropriately with given qargs.
        \param qargs Query arguments
        \param cf An object with operator(NeighborBond) as input.
    */
    template<typename Func>
    void accumulateGeneral(const locality::NeighborQuery* neighbor_query, 
                           const vec3<float>* query_points, unsigned int n_query_points,
                           const locality::NeighborList* nlist,
                           freud::locality::QueryArgs qargs,
                           Func cf)
    {
        m_box = neighbor_query->getBox();
        locality::loopOverNeighbors(neighbor_query, query_points, n_query_points, qargs, nlist, cf);
        m_frame_counter++;
        m_n_points = neighbor_query->getNPoints();
        m_n_query_points = n_query_points;
        // flag to reduce
        m_reduce = true;
    }

protected:
    box::Box m_box;
    unsigned int m_frame_counter; //!< Number of frames calculated
    unsigned int m_n_points;         //!< The number of points
    unsigned int m_n_query_points;           //!< The number of query points
    bool m_reduce;                //!< Whether or not the PCF needs to be reduced

    std::shared_ptr<float> m_pcf_array;         //!< Array of computed pair correlation function
    std::shared_ptr<unsigned int> m_bin_counts; //!< Counts for each bin
    util::ThreadStorage<unsigned int> m_local_bin_counts;
    //!< Thread local bin counts for TBB parallelism
};

}; }; // namespace freud::util

#endif
