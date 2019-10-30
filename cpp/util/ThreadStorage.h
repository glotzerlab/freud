#ifndef THREADSTORAGE_H
#define THREADSTORAGE_H

#include "ManagedArray.h"
#include <tbb/tbb.h>
#include <vector>

namespace freud { namespace util {

//! Wrapper class for enumerable_thread_specific<T*>
/*! It is expected that default value for T is 0.
 */
template<typename T> class ThreadStorage
{
public:
    //! Default constructor
    ThreadStorage()
        : arrays(tbb::enumerable_thread_specific<ManagedArray<T>>([]() { return ManagedArray<T>(); }))
    {}

    //! Constructor with specific size for thread local arrays
    /*! \param size Size of the thread local arrays
     */
    ThreadStorage(size_t size) : ThreadStorage(std::vector<size_t> {size}) {}

    //! Constructor with specific shape for thread local arrays
    /*! \param shape Vector of sizes in each dimension of the thread local arrays
     */
    ThreadStorage(std::vector<size_t> shape)
        : arrays(
            tbb::enumerable_thread_specific<ManagedArray<T>>([shape]() { return ManagedArray<T>(shape); }))
    {}

    //! Destructor
    ~ThreadStorage() {}

    //! Update size of the thread local arrays
    /*! \param size New size of the thread local arrays
     */
    void resize(size_t size)
    {
        resize(std::vector<size_t> {size});
    }

    //! Update size of the thread local arrays
    /*! \param size New size of the thread local arrays
     */
    void resize(std::vector<size_t> shape)
    {
        arrays
            = tbb::enumerable_thread_specific<ManagedArray<T>>([shape]() { return ManagedArray<T>(shape); });
    }

    //! Reset the contents of thread local arrays to be 0
    void reset()
    {
        for (auto array = arrays.begin(); array != arrays.end(); ++array)
        {
            array->reset();
        }
    }

    typedef typename tbb::enumerable_thread_specific<ManagedArray<T>>::const_iterator const_iterator;
    typedef typename tbb::enumerable_thread_specific<ManagedArray<T>>::iterator iterator;
    typedef typename tbb::enumerable_thread_specific<ManagedArray<T>>::reference reference;

    const_iterator begin() const
    {
        return arrays.begin();
    }

    iterator begin()
    {
        return arrays.begin();
    }

    const_iterator end() const
    {
        return arrays.end();
    }

    iterator end()
    {
        return arrays.end();
    }

    reference local()
    {
        return arrays.local();
    }

private:
    tbb::enumerable_thread_specific<ManagedArray<T>> arrays; //!< thread local arrays
};

}; }; // end namespace freud::util

#endif
