#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <vector>
#include <memory>
#ifdef __SSE2__
#include <emmintrin.h>
#endif
#include <sstream>
#include <tbb/tbb.h>
#include <utility>

#include "ManagedArray.h"
#include "utils.h"

namespace freud { namespace util {

//! Class defining an axis of a histogram.
/*! An Axis is defined by a specified number of bins and the boundaries
 * defining them. Given a value along the Axis, the Axis can compute the bin
 * within which this value falls.
 */
class Axis
{
public:
    Axis() : m_nbins(0) {}

    virtual ~Axis() {}

    Axis(size_t nbins, float min, float max) : m_nbins(nbins), m_min(min), m_max(max) {}

    // Return the number of bins in the histogram.
    size_t size() const
    {
        return m_nbins;
    }

    //! Find the bin of a value along this axis.
    /*! This method must be implemented by specific types of axes and is used
     * by the Histogram to bin along each axis independently.
     *
     * \param value The value to bin
     *
     * \return The index of the bin the value falls into.
     */
    virtual size_t bin(const float &value) const = 0;

    //! Return the boundaries of bins.
    std::vector<float> getBinEdges() const
    {
        return m_bin_edges;
    }

    //! Return the centers of bins.
    std::vector<float> getBinCenters() const
    {
        std::vector<float> bin_centers(m_nbins);
        for (unsigned int i = 0; i < m_nbins; i++)
        {
            bin_centers[i] = (m_bin_edges[i] + m_bin_edges[i+1])/float(2.0);
        }
        return bin_centers;
    }

    float getMin() const
    {
        return m_min;
    }

    float getMax() const
    {
        return m_max;
    }

    static const size_t OVERFLOW_BIN = 0xffffffff;

protected:
    size_t m_nbins; //!< Number of bins
    float m_min; //!< Lowest value allowed.
    float m_max; //!< Highest value allowed.
    std::vector<float> m_bin_edges;   //!< The edges of bins.
};

//! A regularly spaced axis.
/*! A RegularAxis is the most common type of axis, representing a series of
 * linearly spaced bins between two boundaries. These axes can be specified a
 * relatively small set of parameter and are very efficient to bin with
 * defining them. Given a value along the Axis, the Axis can compute the bin
 * within which this value falls.
 */
class RegularAxis : public Axis
{
public:
    RegularAxis(size_t nbins, float min, float max) : Axis(nbins, min, max)
    {
        m_bin_edges.resize(m_nbins+1);
        m_dr = (max-min)/static_cast<float>(m_nbins);
        m_dr_inv = 1/m_dr;
        float cur_location = min;
        // This must be <= because there is one extra bin boundary than the number of bins.
        for (unsigned int i = 0; i <= nbins; i++)
        {
            m_bin_edges[i] = (cur_location);
            cur_location += m_dr;
        }
    }

    virtual ~RegularAxis() {}

    //! Find the bin of a value along this axis.
    /*! The linear spacing allows the binning process to be computed especially
     * efficiently since it simply reduces to scaling the value into the range
     * of the axis and looking at just the integral component of the resulting
     * value.
     *
     * \param value The value to bin
     *
     * \return The index of the bin the value falls into.
     */
    virtual size_t bin(const float &value) const
    {
        // Since we're using an unsigned int cast for truncation, we must
        // ensure that we will be working with a positive number or we will
        // fail to detect underflow.
        if (value < m_min || value >= m_max)
        {
            return OVERFLOW_BIN;
        }
        float val = (value - m_min) * m_dr_inv;
        // fast float to int conversion with truncation
#ifdef __SSE2__
        unsigned int bin = _mm_cvtt_ss2si(_mm_load_ss(&val));
#else
        unsigned int bin = (unsigned int)(val);
#endif
        // Avoid rounding leading to overflow.
        if (bin == m_max)
            return bin - 1;
        else
            return bin;
    }

protected:
    float m_dr; //!< Gap between bins
    float m_dr_inv; //!< Inverse gap between bins
};


//! An n-dimensional histogram class.
/*! The Histogram is designed to simplify the most common use of histograms in
 * C++ code, which is looping over a series of values and then binning them. To
 * facilitate this use-case, the class provides an overriden operator() that
 * accepts a set of D values (for a histogram of dimensionality D), computes
 * the bin that value falls into, and then increments the count. Client code
 * can also directly request to know what bin a value would occupy if they wish
 * to operate on the histogram directly. The underlying data is handled using a
 * ManagedArray, allowing dispatch of the multi-dimensional indexing.
 */
class Histogram
{
public:
    //! A container for thread-local copies of a provided histogram.
    /*! This container implements the simplest method of enabling parallel-safe
     * accumulation, namely the creation of separate instances on each thread.
     * Thread local histograms can be accumulated later using the
     * reduceOverThreads functions in the Histogram class.
     *
     * To simplify the implementation and avoid unnecessary copies, the thread
     * local copies all share the same axes (because the axes are stored as
     * arrays of shared_ptrs in the Histogram class). This should cause no
     * problems, but can be refactored if needed.
     */
    class ThreadLocalHistogram
    {
    public:
        ThreadLocalHistogram() {}

        ThreadLocalHistogram(Histogram histogram) : m_local_histograms([histogram]() { return Histogram(histogram.m_axes); }) {}

        typedef typename tbb::enumerable_thread_specific<Histogram>::const_iterator const_iterator;
        typedef typename tbb::enumerable_thread_specific<Histogram>::iterator iterator;
        typedef typename tbb::enumerable_thread_specific<Histogram>::reference reference;

        const_iterator begin() const
        {
            return m_local_histograms.begin();
        }

        iterator begin()
        {
            return m_local_histograms.begin();
        }

        const_iterator end() const
        {
            return m_local_histograms.end();
        }

        iterator end()
        {
            return m_local_histograms.end();
        }

        reference local()
        {
            return m_local_histograms.local();
        }

        void reset()
        {
            for (auto hist = m_local_histograms.begin(); hist != m_local_histograms.end(); ++hist)
            {
                hist->reset();
            }
        }

        //! Dispatch to thread local histogram.
        template <typename ... Floats>
        void operator()(Floats ... values)
        {
            m_local_histograms.local()(values ...);
        }

    protected:
        tbb::enumerable_thread_specific<Histogram> m_local_histograms;  //!< The thread-local copies of m_histogram.
    };

    typedef std::vector<std::shared_ptr<Axis> > Axes;
    typedef Axes::const_iterator AxisIterator;

    //! Default constructor
    Histogram() {}

    //! Constructor
    Histogram(std::vector<std::shared_ptr<Axis>> axes) : m_axes(axes)
    {
        std::vector<unsigned int> sizes;
        for (AxisIterator it = m_axes.begin(); it != m_axes.end(); it++)
            sizes.push_back((*it)->size());
        m_bin_counts = ManagedArray<unsigned int>(sizes);
    }

    //! Destructor
    ~Histogram() {};

    //! Bin value and update the histogram count.
    template <typename ... Floats>
    void operator()(Floats ... values)
    {
        std::vector<float> value_vector = getValueVector(values ...);
        size_t value_bin = bin(value_vector);
        // Check for sentinel to avoid overflow.
        if (value_bin != Axis::OVERFLOW_BIN)
        {
            m_bin_counts[value_bin]++; // TODO: Will want to replace this with custom accumulation at some point.
        }
    }

    //! Find the bin of a value.
    /*! Bins are first computed along each axis of the histogram. These bins
     *  are then combined into a single linear index using the underlying
     *  ManagedArray.
     */
    size_t bin(std::vector<float> values) const
    {
        if (values.size() != m_axes.size())
        {
            std::ostringstream msg;
            msg << "This Histogram is " << m_axes.size() << "-dimensional, but only " << values.size() << " values were provided in bin" << std::endl;
            throw std::invalid_argument(msg.str());
        }
        // First bin the values along each axis.
        std::vector<unsigned int> ax_bins;
        for (unsigned int ax_idx = 0; ax_idx < m_axes.size(); ax_idx++)
        {
            unsigned int bin_i = m_axes[ax_idx]->bin(values[ax_idx]);
            // Immediately return sentinel if any bin is out of bounds.
            if (bin_i == Axis::OVERFLOW_BIN)
            {
                return Axis::OVERFLOW_BIN;
            }
            ax_bins.push_back(bin_i);
        }

        return m_bin_counts.getIndex(ax_bins);
    }

    //! Get the computed histogram.
    const ManagedArray<unsigned int> &getBinCounts()
    {
        return m_bin_counts;
    }

    //! Get the shape of the computed histogram.
    std::vector<unsigned int> shape() const
    {
        return m_bin_counts.shape();
    }

    //! Reset the histogram.
    void reset()
    {
        m_bin_counts.reset();
    }

    //! Return the edges of bins.
    /*! This vector will be of size axis.size()+1 for each axis.
     */
    std::vector<std::vector<float> > getBinEdges() const
    {
        std::vector<std::vector<float> > bins(m_axes.size());
        for (unsigned int i = 0; i < m_axes.size(); ++i)
        {
            bins[i] = m_axes[i]->getBinEdges();
        }
        return bins;
    }

    //! Return the bin centers.
    /*! This vector will be of size axis.size() for each axis.
     */
    std::vector<std::vector<float> > getBinCenters() const
    {
        std::vector<std::vector<float> > bins(m_axes.size());
        for (unsigned int i = 0; i < m_axes.size(); ++i)
        {
            bins[i] = m_axes[i]->getBinCenters();
        }
        return bins;
    }

    //! Return a vector of tuples (min, max) indicating the bounds of each axis.
    std::vector<std::pair<float, float> > getBounds() const
    {
        std::vector<std::pair<float, float> > bounds(m_axes.size());
        for (unsigned int i = 0; i < m_axes.size(); ++i)
        {
            bounds[i] = std::pair<float, float>(
                m_axes[i]->getMin(),
                m_axes[i]->getMax()
                );
        }
        return bounds;
    }

    //! Return a vector indicating the number of bins in each axis.
    std::vector<unsigned int> getAxisSizes() const
    {
        std::vector<unsigned int> sizes(m_axes.size());
        for (unsigned int i = 0; i < m_axes.size(); ++i)
        {
            sizes[i] = m_axes[i]->size();
        }
        return sizes;
    }

    //!< Aggregate a set of thread-local histograms into this one and apply a function.
    /*! This function can be used whenever reduction over a set of
     * ThreadLocalHistograms requires additional post-processing, such as some
     * sort of normalization per bin.
     *
     * \param local_histograms The set of local histograms to reduce into this one.
     * \param cf The function to apply to each bin, must have signature (size_t i) {...}
     */
    template <typename ComputeFunction>
    void reduceOverThreadsPerBin(ThreadLocalHistogram &local_histograms, const ComputeFunction &cf)
    {
        util::forLoopWrapper(0, m_bin_counts.size(), [=](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i)
            {
                for (ThreadLocalHistogram::const_iterator local_bins = local_histograms.begin();
                    local_bins != local_histograms.end(); ++local_bins)
                {
                    m_bin_counts[i] += (*local_bins).m_bin_counts[i];
                }
                
                cf(i);
            }
        });
    }

    //!< Aggregate a set of thread-local histograms into this one.
    /*! This function is the standard method for parallel aggregation of a
     * histogram. The simplest way to achieve parallel-safe accumulation is to
     * create a ThreadLocalHistogram object, which holds a local copy of a
     * histogram on each thread. This function then accumulates the results
     * into this object.
     *
     * \param local_histograms The set of local histograms to reduce into this one.
     */
    void reduceOverThreads(ThreadLocalHistogram &local_histograms)
    {
        // Simply call the per-bin function with a nullary function.
        reduceOverThreadsPerBin(local_histograms, [](size_t i) {});
    }

    //! Read-only index into array.
    unsigned int &operator[](unsigned int i)
    {
        return m_bin_counts[i];
    }

    //! Read-only index into array.
    const unsigned int &operator[](unsigned int i) const
    {
        return m_bin_counts[i];
    }

    unsigned int size() const
    {
        return m_bin_counts.size();
    }

protected:
    std::vector<std::shared_ptr<Axis > > m_axes; //!< The axes.
    ManagedArray<unsigned int> m_bin_counts; //!< Counts for each bin

    //! The base case for constructing a vector of values provided to operator().
    /*! This function and the accompanying recursive function below employ
     * variadic templating to accept an arbitrary set of float values and
     * construct a vector out of them.
     */
    std::vector<float> getValueVector(float value) const
    {
        return {value};
    }

    //! The recursive case for constructing a vector of values (see base-case function docs).
    template <typename ... Floats>
    std::vector<float> getValueVector(float value, Floats ... values) const
    {
        std::vector<float> tmp = getValueVector(values...);
        tmp.insert(tmp.begin(), value);
        return tmp;
    }
};

}; }; // namespace freud::util

#endif
