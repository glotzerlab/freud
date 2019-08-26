#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <vector>
#include <memory>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

namespace freud { namespace util {

////! Index an N-dimensional array.
/*! Row major mapping of N-dimensional mapping to 1D.
*/
//class IndexND
//{
//public:
    ////! Constructor
    /*! \param w Width of the cubic 3D array
     */
    //inline IndexND(unsigned int w = 0) : m_w(w), m_h(w), m_d(w) {}

    ////! Contstructor
    //[>! \param w Width of the 3D array
        //\param h Height of the 3D array
        //\param d Depth of the 3D array
    //*/
    //inline IndexND(unsigned int w, unsigned int h, unsigned int d) : m_w(w), m_h(h), m_d(d) {}

    ////! Calculate an index
    //[>! \param i index along the width
        //\param j index up the height
        //\param k index along the depth
        //\returns 1D array index corresponding to the 3D index (\a i, \a j, \a k) in row major order
    //*/
    //inline unsigned int operator()(unsigned int i, unsigned int j, unsigned int k) const
    //{
        //return k * m_w * m_h + j * m_w + i;
    //}

    //////! Unravel an index
    ////[>! \param i 1D index along the width
        ////\returns 3D index (\a i, \a j, \a k) corresponding to the 1D index (\a i) in row major order
    ///[>/
    ////vec3<unsigned int> operator()(unsigned int i) const
    ////{
        ////vec3<unsigned int> l_idx;
        ////l_idx.x = i % m_w;
        ////l_idx.y = (i / m_w) % m_h;
        ////l_idx.z = i / (m_w * m_h);
        ////return l_idx;
    ////}

    ////! Get the number of 1D elements stored
    /*! \returns Number of elements stored in the underlying 1D array
     */
    //inline unsigned int getNumElements() const
    //{
        //return m_w * m_h * m_d;
    //}

    ////! Get the width of the 3D array
    //inline unsigned int getW() const
    //{
        //return m_w;
    //}

    ////! Get the height of the 3D array
    //inline unsigned int getH() const
    //{
        //return m_h;
    //}

    ////! Get the depth of the 3D array
    //inline unsigned int getD() const
    //{
        //return m_d;
    //}


//private:
    //unsigned int m_w; //!< Width of the 3D array
    //unsigned int m_h; //!< Height of the 3D array
    //unsigned int m_d; //!< Depth of the 3D array
//};




// Class defining an axis of an array (used in histogram).
// T is the data type of the axis
class Axis
{
public:
    Axis() : m_nbins(0) {}

    // How many bins are there in this axis?
    size_t size() const
    {
        return m_nbins;
    }

    // What bin is the value in.
    virtual size_t getBin(const float &value) const = 0;

protected:
    size_t m_nbins; //!< Number of bins
};

// A regularly spaced axis
class RegularAxis : public Axis
{
public:
    RegularAxis(size_t nbins, float min, float max) : Axis(), m_min(min), m_max(max)
    {
        this->m_nbins = nbins;
        m_dr = (max-min)/nbins;
        float cur_location = min + m_dr/2;
        for (unsigned int i = 0; i < nbins; i++)
        {
            m_bins.push_back(cur_location);
            cur_location += m_dr;
        }
    }

    virtual size_t getBin(const float &value) const
    {
        // fast float to int conversion with truncation
#ifdef __SSE2__
        unsigned int bin = _mm_cvtt_ss2si(_mm_load_ss(&value));
#else
        unsigned int bin = (unsigned int)(value);
#endif
        return bin;
    }

protected:
    std::vector<float> m_bins;   //!< Bin locations.
    float m_min; //!< Lowest value allowed.
    float m_max; //!< Highest value allowed.
    float m_dr; //!< Gap between bins
};

template<typename T> std::shared_ptr<T> makeEmptyArray(unsigned int size)
{
    auto new_arr = std::shared_ptr<T>(new T[size], std::default_delete<T[]>());
    memset((void*) new_arr.get(), 0, sizeof(T) * size);
    return new_arr;
}

class Histogram
{
public:
    //! Constructor
    Histogram(size_t nbins, float min, float max)
    {
        m_axes.push_back(std::make_shared<RegularAxis>(nbins, min, max));
        m_bin_counts = makeEmptyArray<unsigned int>(nbins);
    }

    //! Destructor
    virtual ~Histogram() {};

/*! Determine the bin of a value and increment it.
 *  This operator
 */
    void operator()(float value)
    {
        size_t bin = getBin(std::vector<float> {value});
        m_bin_counts.get()[bin]++; // Will want to replace this with custom accumulation at some point.
    }

    size_t getBin(std::vector<float> values)
    {
        // First bin the values along each axis.
        std::vector<size_t> ax_bins;
        for (unsigned int ax_idx = 0; ax_idx < m_axes.size(); ax_idx++)
        {
            ax_bins.push_back(m_axes[ax_idx]->getBin(values[ax_idx]));
        }

        // Now get the corresponding linear bin.
        size_t cur_prod = 1;
        size_t idx = 0;
        // We must iterate over bins in reverse order to build up the value of
        // prod because each subsequent axis contributes less according to
        // row-major ordering.
        for (int ax_idx = m_axes.size() - 1; ax_idx >= 0; ax_idx--)
        {
            idx += ax_bins[ax_idx] * cur_prod;
            cur_prod *= m_axes[ax_idx]->size();
        }
        return idx;
    }

    std::shared_ptr<unsigned int> m_bin_counts; //!< Counts for each bin

protected:
    std::vector<std::shared_ptr<Axis > > m_axes; //!< The axes.
};

}; }; // namespace freud::util

#endif
