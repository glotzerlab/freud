// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef MANAGED_ARRAY_H
#define MANAGED_ARRAY_H

#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>

#include "VectorMath.h"

/*! \file ManagedArray.h
    \brief Defines the standard array class to be used throughout freud.
*/

namespace freud { namespace util {

//! Class to handle the storage of all arrays of numerical data used in freud.
/*! The purpose of this class is to handle standard memory management, and to
 *  provide an abstraction around the implementation-specific choice of
 *  underlying data structure for arrays of data in freud. These arrays are
 *  specifically designed for numerical data calculated by a compute class,
 *  particularly for arrays that must be made accessible through the Python API
 *  of freud. The array shape is stored and used to support multidimensional
 *  indexing.
 *
 *  Performance notes:
 *      1. The variadic indexers may be a bottleneck if used in
 *         performance-critical code paths. In such cases, directly calling the
 *         overloaded signature using an std::vector is preferable.
 *      2. In situations where multiple identically shaped arrays are being
 *         indexed into, the index may be computed once using the getIndex
 *         function and reused to avoid recomputing it each time.
 */
template<typename T, size_t Ndim> class ManagedArray
{
public:
    //! Constructor based on a shape tuple.
    /*! Including a default value for the shape allows the usage of this
     *  constructor as the default constructor.
     *
     *  \param shape Shape of the array to allocate.
     */
    explicit ManagedArray(const std::array<size_t, Ndim>& shape = {0}) : m_shape(shape)
    {
        m_size = 1;
        # pragma unroll
        for (unsigned int i = 0; i < Ndim; ++i)
        {
            m_size *= m_shape[i];
        }

        m_data = std::vector<T>(size());
        reset();
    }

    //! Constructor based on a shape tuple.
    /*! Including a default value for the shape allows the usage of this
     *  constructor as the default constructor.
     *
     *  \param shape Shape of the array to allocate.
     */
    explicit ManagedArray(size_t size) : ManagedArray(std::array<size_t, 1> {size}) {}

    //! Destructor (currently empty because data is managed by shared pointer).
    ~ManagedArray() = default;

    //! Reset the contents of array to be 0.
    void reset()
    {
        if (size() != 0)
        {
            memset((void*) get(), 0, sizeof(T) * size());
        }
    }

    //! Return a constant pointer to the underlying data
    const T* data() const
    {
        return m_data.data();
    }

    //! Return the underlying pointer (requires two levels of indirection).
    /*! This function should only be used by client code when a raw pointer is
     * absolutely required. It is primarily part of the public API for the
     * purpose of freud's Python API, which requires a non-const pointer to the
     * data to construct a numpy array. There are specific use-cases (e.g.
     * interacting with Eigen) where directly accessing a mutable version of
     * the underlying data pointer is necessary, but users should be cautious
     * about overusing calls to get() rather than using the various operators
     * provided by the class.
     */
    T* data()
    {
        return m_data.data();
    }

    //! Writeable index into array.
    T& operator[](size_t index)
    {
        if (index >= size())
        {
            std::ostringstream msg;
            msg << "Attempted to access index " << index << " in an array of size " << size() << std::endl;
            throw std::invalid_argument(msg.str());
        }
        return m_data[index];
    }

    //! Read-only index into array.
    const T& operator[](size_t index) const
    {
        if (index >= size())
        {
            std::ostringstream msg;
            msg << "Attempted to access index " << index << " in an array of size " << size() << std::endl;
            throw std::invalid_argument(msg.str());
        }
        return m_data[index];
    }

    //! Get the size of the current array.
    size_t size() const
    {
        return m_size;
    }

    //! Get the shape of the current array.
    std::array<size_t, Ndim> shape() const
    {
        return m_shape;
    }

    //*************************************************************************
    // In order to support convenient indexing using arbitrary numbers of
    // indices, we provide overloads of the indexing operator.  All calls
    // eventually funnel through the simplest function interface, a std::vector
    // of indices. However, the more convenient approach is enabled using
    // variadic template arguments.
    //
    // Performance note: the variadic indexers can be expensive
    // since they have to build up a std::vector through a sequence of
    // (hopefully compiler optimized) function calls. In general, this is
    // unimportant since performance critical code paths in freud do not
    // involve large amounts of indexing into ManagedArrays. However, in cases
    // where critical code paths do index into the array, it may be
    // advantageous to directly call the function with a std::vector of indices
    // and bypass the variadic indexers.
    //*************************************************************************

    //! Implementation of variadic indexing function.
    template<typename... Ints> inline T& operator()(Ints... indices)
    {
        // cppcheck generates a false positive here on old machines (CI),
        // probably due to limited template support on those compilers.
        // cppcheck-suppress returnTempReference
        return (*this)(std::array<size_t, Ndim> {indices});
    }

    //! Constant implementation of variadic indexing function.
    template<typename... Ints> inline const T& operator()(Ints... indices) const
    {
        // cppcheck generates a false positive here on old machines (CI),
        // probably due to limited template support on those compilers.
        // cppcheck-suppress returnTempReference
        return (*this)(std::array<size_t, Ndim> {indices});
    }

    //! Core function for multidimensional indexing.
    /*! All the other convenience functions for indexing ultimately call this
     * function (or the const version below), which operates on a vector of
     * indexes.
     *
     * Note that the logic in getIndex is intentionally inlined here for
     * performance reasons. Although unimportant in most cases, operator() can
     * become a performance bottleneck when used in highly performance critical
     * code paths.
     */
    inline T& operator()(const std::array<size_t, Ndim>& indices)
    {
        size_t cur_prod = 1;
        size_t idx = 0;
        // In getting the linear bin, we must iterate over bins in reverse
        // order to build up the value of cur_prod because each subsequent axis
        // contributes less according to row-major ordering.
        # pragma unroll
        for (unsigned int i = Ndim - 1; i != static_cast<unsigned int>(-1); --i)
        {
            idx += indices[i] * cur_prod;
            cur_prod *= m_shape[i];
        }
        return (*this)[idx];
    }

    //! Const version of core function for multidimensional indexing.
    inline const T& operator()(const std::array<size_t, Ndim>& indices) const
    {
        size_t cur_prod = 1;
        size_t idx = 0;
        // In getting the linear bin, we must iterate over bins in reverse
        // order to build up the value of cur_prod because each subsequent axis
        // contributes less according to row-major ordering.
        # pragma unroll
        for (unsigned int i = Ndim - 1; i != static_cast<unsigned int>(-1); --i)
        {
            idx += indices[i] * cur_prod;
            cur_prod *= m_shape[i];
        }
        return (*this)[idx];
    }

    //! Get the multi-index corresponding to a single regular index.
    /*! This function is provided as an external utility in the event that
     * index generation is necessary without an actual array.
     *
     *  \param shape The shape to map indexes to.
     *  \param indices The index in each dimension.
     */
    static inline std::array<size_t, Ndim> getMultiIndex(const std::array<size_t, Ndim>& shape, size_t index)
    {
        size_t index_size = std::accumulate(shape.cbegin(), shape.cend(), 1, std::multiplies<>());

        std::array<size_t, Ndim> indices(shape.size());

        # pragma unroll
        for (unsigned int i = 0; i < Ndim; ++i)
        {
            index_size /= shape[i];
            // Integer division should cast away extras.
            indices[i] = index / index_size;
            index %= index_size;
        }
        return indices;
    }

    //! Get the linear index corresponding to a vector of indices in each dimension.
    /*! This function is provided as an external utility in the event that
     * index generation is necessary without an actual array.
     *
     *  \param shape The shape to map indexes to.
     *  \param indices The index in each dimension.
     */
    static inline size_t getIndex(const std::array<size_t, Ndim>& shape, const std::array<size_t, Ndim>& indices)
    {
        size_t cur_prod = 1;
        size_t idx = 0;
        // In getting the linear bin, we must iterate over bins in reverse
        // order to build up the value of cur_prod because each subsequent axis
        // contributes less according to row-major ordering.
        # pragma unroll
        for (unsigned int i = Ndim - 1; i != static_cast<unsigned int>(-1); --i)
        {
            idx += indices[i] * cur_prod;
            cur_prod *= shape[i];
        }
        return idx;
    }

    //! Get the linear index corresponding to a vector of indices in each dimension.
    /*! This function performs the conversion of a vector of indices
     *  into a linear index into the underlying data array of the ManagedArray.
     *  This function is primarily provided as a convenience, but may be useful
     *  to generate an index that can be reused multiple times.
     *
     *  \param indices The index in each dimension.
     */
    inline size_t getIndex(const std::array<size_t, Ndim>& indices) const
    {
        # pragma unroll
        for (unsigned int i = 0; i < Ndim; ++i)
        {
            if (indices[i] > m_shape[i])
            {
                std::ostringstream msg;
                msg << "Attempted to access index " << indices[i] << " in dimension " << i
                    << ", which has size " << m_shape[i] << std::endl;
                throw std::invalid_argument(msg.str());
            }
        }

        return getIndex(m_shape, indices);
    }

    //! Return a copy of this array.
    /*! The returned object is a deep copy in the sense that it will copy every
     * element of the stored array. However, if the stored elements are
     * themselves pointers (e.g. if you create a ManagedArray<int*>), then the
     * copy will also retain pointers to that data.
     */
    ManagedArray copy() const
    {
        ManagedArray newarray(shape());
        for (unsigned int i = 0; i < size(); ++i)
        {
            newarray[i] = m_data[i];
        }
        return newarray;
    }

private:
    std::vector<T> m_data;            //!< array data.
    std::array<size_t, Ndim> m_shape; //!< Shape of array.
    size_t m_size;                    //!< number of array elements.
};

// explicit instantiations for python extension modules
template class ManagedArray<float, 1>;
template class ManagedArray<double, 1>;
template class ManagedArray<unsigned int, 1>;
template class ManagedArray<vec3<float>, 1>;

}; }; // end namespace freud::util

#endif
