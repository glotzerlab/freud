#ifndef HISTOGRAM_COMPUTE_H
#define HISTOGRAM_COMPUTE_H

#include "Histogram.h"
#include "Box.h"
#include "NeighborComputeFunctional.h"
#include "NeighborQuery.h"

namespace freud { namespace locality {

//! Perform parallel histogram computations.
/*! The BondHistogramCompute class serves as a parent class for freud computes
 * that compute histograms of neighbor bonds. It encapsulates a Histogram
 * instance as well as the thread local copies and accumulation and reduction
 * over these copies. It also offers some generalized functionality for process
 * of accumulating histograms over many frames, assuming that computations must
 * be performed on a per-NeighborBond basis. 
*/
class BondHistogramCompute
{
public:
    //! Default constructor
    BondHistogramCompute() : m_box(box::Box()), m_frame_counter(0), m_n_points(0), m_n_query_points(0), m_reduce(true), m_histogram(), m_local_histograms() {}

    //! Destructor
    virtual ~BondHistogramCompute() {};

    //! Reset the RDF array to all zeros
    void reset()
    {
        m_local_histograms.reset();
        this->m_frame_counter = 0;
        this->m_reduce = true;
    }

    //! Reduce thread-local arrays onto the primary data arrays.
     virtual void reduce() = 0;

    //! Get the simulation box
    const box::Box& getBox() const
    {
        return m_box;
    }

    //! Return :code:`thing_to_return` after reducing.
    template<typename U>
    U &reduceAndReturn(U &thing_to_return)
    {
        if (m_reduce == true)
        {
            reduce();
        }
        m_reduce = false;
        return thing_to_return;
    }

    //! Get a reference to the bin counts array
    const util::ManagedArray<unsigned int> &getBinCounts()
    {
        return reduceAndReturn(m_histogram.getBinCounts());
    }

    //! Get bin centers.
    std::vector<std::vector<float> > getBinCenters() const
    {
        return m_histogram.getBinCenters();
    }

    //! Return the bin boundaries.
    std::vector<std::vector<float> > getBinEdges() const
    {
        return m_histogram.getBinEdges();
    }

    //! Return the bin boundaries.
    std::vector<std::pair<float, float> > getBounds() const
    {
        return m_histogram.getBounds();
    }

    //! Return the bin boundaries.
    std::vector<unsigned int> getAxisSizes() const
    {
        return m_histogram.getAxisSizes();
    }

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
    unsigned int m_frame_counter;            //!< Number of frames calculated.
    unsigned int m_n_points;                 //!< The number of points.
    unsigned int m_n_query_points;           //!< The number of query points.
    bool m_reduce;                           //!< Whether or not the histogram needs to be reduced.

    util::Histogram<unsigned int> m_histogram;             //!< Histogram of interparticle distances (bond lengths).
    util::Histogram<unsigned int>::ThreadLocalHistogram m_local_histograms;   //!< Thread local bin counts for TBB parallelism

    typedef util::Histogram<unsigned int> BondHistogram;
    typedef BondHistogram::Axes BHAxes;
};

}; }; // namespace freud::util

#endif  // HISTOGRAM_COMPUTE_H
