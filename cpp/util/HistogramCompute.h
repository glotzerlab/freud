#ifndef HISTOGRAM_COMPUTE_H
#define HISTOGRAM_COMPUTE_H

#include "Histogram.h"
#include "Box.h"

namespace freud { namespace util {

//! Perform parallel histogram computations.
/*! The HistogramCompute class serves as a parent class for freud computes that
 * use a histogram. It encapsulates a Histogram instance as well as the thread
 * local copies and accumulation and reduction over these copies.
*/
class HistogramCompute
{
public:
    //! Default constructor
    HistogramCompute() : m_box(box::Box()), m_frame_counter(0), m_n_points(0), m_n_query_points(0), m_reduce(true) {}

    //! Destructor
    virtual ~HistogramCompute() {};

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
    template<typename T>
    T &reduceAndReturn(T &thing_to_return)
    {
        if (m_reduce == true)
        {
            reduce();
        }
        m_reduce = false;
        return thing_to_return;
    }


protected:
    box::Box m_box;
    unsigned int m_frame_counter;            //!< Number of frames calculated.
    unsigned int m_n_points;                 //!< The number of points.
    unsigned int m_n_query_points;           //!< The number of query points.
    bool m_reduce;                           //!< Whether or not the histogram needs to be reduced.

    util::Histogram m_histogram;             //!< Histogram of interparticle distances (bond lengths).
    util::Histogram::ThreadLocalHistogram m_local_histograms;   //!< Thread local bin counts for TBB parallelism
};

}; }; // namespace freud::util

#endif  // HISTOGRAM_COMPUTE_H
