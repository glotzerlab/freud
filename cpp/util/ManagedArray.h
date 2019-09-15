#ifndef MANAGED_ARRAY_H
#define MANAGED_ARRAY_H

#include <memory>
#include <vector>
#include <sstream>

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
 *  To support resizing, a ManagedArray instances stores its data as a pointer
 *  to a pointer, and its shape as a pointer. As a result, copy-assignment or
 *  initialization will result in a new ManagedArray pointing to the same data,
 *  and any such array can resize or reallocate this data. The pointer to
 *  pointer infrastructure ensures that such changes properly propagate to all
 *  ManagedArrays referencing a given memory space. In addition, this
 *  infrastructure allows the creation of a completely new ManagedArray with a
 *  new set of pointers that also manages the same data, allowing it to keep
 *  the original array alive if the original ManagedArray instances become
 *  decoupled from it.
 *
 *  Performance notes:
 *      1. The variadic indexers may be a bottleneck if used in
 *         performance-critical code paths. In such cases, directly calling the
 *         overloaded signature using an std::vector is preferable.
 *      2. In situations where multiple identically shaped arrays are being
 *         indexed into, the index may be computed once using the getIndex
 *         function and reused to avoid recomputing it each time.
 */
template<typename T> class ManagedArray
{
public:
    //! Constructor based on a shape tuple.
    /*! Including a default value for the shape allows the usage of this
     *  constructor as the default constructor.
     *
     *  \param shape Shape of the array to allocate.
     */
    ManagedArray(std::vector<unsigned int> shape = {0})
    {
        prepare(shape, true);
    }

    //! Destructor (currently empty because data is managed by shared pointer).
    ~ManagedArray() {}

    //! Simple convenience for 1D arrays that calls through to the shape based `prepare` function.
    /*! \param new_size Size of the 1D array to allocate.
     */
    void prepare(unsigned int new_size)
    {
        prepare(std::vector<unsigned int> {new_size});
    }

    //! Prepare for writing new data.
    /*! This function always resets the array to contain zeros, but it will
     * also reallocate if there are other ManagedArrays pointing to the data in
     * order to ensure that those array references are not invalidated when
     * this function clears the data.
     *
     *  \param new_shape Shape of the array to allocate.
     *  \param force Reallocate regardless of whether anything changed or needs to be persisted.
     */
    void prepare(std::vector<unsigned int> new_shape, bool force=false)
    {
        // If we resized, or if there are outstanding references, we create a new array. No matter what, reset.
        if (force || (m_data.use_count() > 1) || (new_shape != shape()))
        {
            m_shape = std::make_shared<std::vector<unsigned int> >(new_shape);

            m_size = std::make_shared<unsigned int>(1);
            for (int i = m_shape->size() - 1; i >= 0; --i)
            {
                (*m_size) *= (*m_shape)[i];
            }

            m_data = std::shared_ptr<std::shared_ptr<T> >(
                new std::shared_ptr<T>(new T[size()], std::default_delete<T[]>()));
        }
        reset();
    }

    //! Reset the contents of array to be 0.
    void reset()
    {
        if (size() != 0)
        {
            memset((void*) get(), 0, sizeof(T) * size());
        }
    }

    //! Return the underlying pointer (requires two levels of indirection).
    /*! This function SHOULD NOT BE USED. It is only part of the public API for
     * the purpose of freud's Python API, which requires a non-const pointer to
     * the data to construct a numpy array. However, this pointer to the
     * underlying should never be accessed by client code.
     */
    T *get() const
    {
        std::shared_ptr<T> *tmp = m_data.get();
        return (*tmp).get();
    }

    //! Writeable index into array.
    T &operator[](unsigned int index)
    {
        if (index >= size())
        {
            std::ostringstream msg;
            msg << "Attempted to access index " << index << " in an array of size " << size() << std::endl;
            throw std::invalid_argument(msg.str());
        }
        return get()[index];
    }


    //! Read-only index into array.
    const T &operator[](unsigned int index) const
    {
        if (index >= size())
        {
            std::ostringstream msg;
            msg << "Attempted to access index " << index << " in an array of size " << size() << std::endl;
            throw std::invalid_argument(msg.str());
        }
        return get()[index];
    }

    //! Get the size of the current array.
    unsigned int size() const
    {
        return *m_size;
    }

    //! Get the shape of the current array.
    std::vector<unsigned int> shape() const
    {
        return *m_shape;
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
    template <typename ... Ints>
    inline T &operator()(Ints ... indices)
    {
        return (*this)(buildIndex(indices...));
    }

    //! Constant implementation of variadic indexing function.
    template <typename ... Ints>
    inline const T &operator()(Ints ... indices) const
    {
        return (*this)(buildIndex(indices ...));
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
    inline T &operator()(std::vector<unsigned int> indices)
    {
        size_t cur_prod = 1;
        size_t idx = 0;
        for (int i = indices.size() - 1; i >= 0; --i)
        {
            idx += indices[i] * cur_prod;
            cur_prod *= (*m_shape)[i];
        }
        return (*this)[idx];
    }

    //! Const version of core function for multidimensional indexing.
    inline const T &operator()(std::vector<unsigned int> indices) const
    {
        size_t cur_prod = 1;
        size_t idx = 0;
        for (int i = indices.size() - 1; i >= 0; --i)
        {
            idx += indices[i] * cur_prod;
            cur_prod *= (*m_shape)[i];
        }
        return (*this)[idx];
    }

    //! Get the linear index corresponding to a vector of indices in each dimension.
    /*! This function performs the conversion of a vector of indices
     *  into a linear index into the underlying data array of the ManagedArray.
     *  This function is primarily provided as a convenience, but may be useful
     *  to generate an index that can be reused multiple times.
     *
     *  \param indices The index in each dimension.
     */
    inline size_t getIndex(std::vector<unsigned int> indices) const
    {
        // In getting the linear bin, we must iterate over bins in reverse
        // order to build up the value of cur_prod because each subsequent axis
        // contributes less according to row-major ordering.
        size_t cur_prod = 1;
        size_t idx = 0;
        for (int i = indices.size() - 1; i >= 0; --i)
        {
            idx += indices[i] * cur_prod;
            cur_prod *= (*m_shape)[i];
        }
        return idx;
    }


private:

    //! The base case for building up the index.
    /*! These argument building functions are templated on two types, one that
     *  encapsulates the current object being operated on and the other being
     *  the list of remaining arguments. Since users may provide both signed and
     *  unsigned ints to the function, we perform the appropriate check on each
     *  Int object. The second function is used for template recursion in
     *  unwrapping the list of arguments.
     */
    template <typename Int>
    inline static std::vector<unsigned int> buildIndex(Int index)
    {
        return {static_cast<unsigned int>(index)};
    }

    //! The recursive case for building up the index (see above).
    template <typename Int, typename ... Ints>
    inline static std::vector<unsigned int> buildIndex(Int index, Ints ... indices)
    {
        std::vector<unsigned int> tmp = buildIndex(indices...);
        tmp.insert(tmp.begin(), static_cast<unsigned int>(index));
        return tmp;
    }

    std::shared_ptr<std::shared_ptr<T> > m_data;           //!< Pointer to array.
    std::shared_ptr<std::vector<unsigned int> > m_shape;   //!< Shape of array.
    std::shared_ptr<unsigned int> m_size;                  //!< Size of array.
};

}; }; // end namespace freud::util

#endif
