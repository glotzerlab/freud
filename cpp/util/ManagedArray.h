#ifndef NUMERICAL_ARRAY_H
#define NUMERICAL_ARRAY_H

#include <memory>
#include <tbb/tbb.h>

/*! \file ManagedArray.h
    \brief Defines the standard array class to be used throughout freud.
*/

namespace freud { namespace util {

//! Class to handle the storage of all arrays of numerical data used in freud.
/*! The purpose of this class is to handle standard memory management, and to
 *  provide an abstraction around the implementation-specific choice of
 *  underlying data structure for arrays of data in freud. These arrays are
 *  specifically designed for numerical data, allowing operations such as a
 *  memset to 0 for clearing the data.
 *
 *  A ManagedArray can be in one of three states:
 *      1. Managing its own memory (used for class members).
 *      2. Pointing to data managed by another ManagedArray.
 *      3. Pointing to external data through a pointer (e.g. a NumPy array).
 *
 *  To support resizing, a ManagedArray instances stores its size as a pointer,
 *  allowing reference arrays to point to the size of the original array if it
 *  is resized. In other words, ManagedArray instances in state 2 will always
 *  have the correct size available. However, ManagedArray objects in state 3
 *  have a fixed array size and assume that the original data size is fixed for
 *  the lifetime of the ManagedArray. This restriction allows, for instance,
 *  the = operator to be well-defined in terms of the ownership of both array
 *  data and the size information, which is shared by all views into a given
 *  array if it is owned by some ManagedArray.
 *
 *  While the class provides strong protection against memory leaks or improper
 *  access of data, the user is responsible for ensuring that a ManagedArray
 *  that does not manage its own data does not attempt to access data after the
 *  underlying data has been destructed by the owner. This is generally safe
 *  within the freud C++ API because compute classes in freud are designed to
 *  accept ManagedArrays as inputs, which therefore must be constructed by the
 *  calling code. Within the freud Python API, this logic is handled through
 *  proper usage of the acquisition methods of this class, which allow
 *  ManagedArrays to transfer data ownership. The class is designed around the
 *  expectation of a Python wrapper class that will maintain ownership of a
 *  ManagedArray and its underlying data between compute calls.
 *
 *  There are two specific scenarios when ManagedArrays are invalidated.
 *      1. ManagedArrays in state 2 will be invalidated if the data source
 *         array is deleted.
 *      2. ManagedArrays in state 3 will be invalidated if the data source is
 *         deleted or changes in size.
 *
 *  No attempt is made to protect against these failures. The user is
 *  responsible for avoiding these cases.
 */
template<typename T> class ManagedArray
{
public:
    //! Default constructor with optional size for thread local arrays
    /*! \param size Size of the array to allocate.
     */
    ManagedArray(unsigned int size=0)
    {
        m_size = std::make_shared<unsigned int>(size);
        // Hard code making an array of at least size 1.
        m_data = std::shared_ptr<std::shared_ptr<T> >(
            new std::shared_ptr<T>(new T[1], std::default_delete<T[]>()));
        create_new_array(size);
        reset();
    }

    //! Destructor (currently empty because data is managed by shared pointer).
    ~ManagedArray() {}

    //! Update size of the array.
    /*! \param size New size of the array.
     */
    void resize(unsigned int size)
    {
        if (size != *m_size)
        {
            *m_size = size;
            reallocate();
        }
    }

    //! Allocate new memory for the array.
    /*! This function primarily serves as a way to reallocate new memory from
     *  the Python API if needed.
     * \param size New size of the array.
     */
    void reallocate()
    {
        create_new_array(*m_size);
        reset();
    }

    //! Reset the contents of array to be 0.
    void reset()
    {
        if (*m_size != 0)
        {
            memset((void*) get(), 0, sizeof(T) * (*m_size));
        }
    }

    //! Return the underlying pointer (requires two levels of indirection).
    T *get() const
    {
        std::shared_ptr<T> * tmp = m_data.get();
        return (*tmp).get();
    }

    //! Writeable index into array.
    T &operator[](unsigned int index)
    {
        if (index >= *m_size)
        {
            throw std::runtime_error("Attempted to access data out of bounds.");
        }
        return get()[index];
    }

    //! Read-only index into array.
    const T &operator[](unsigned int index) const
    {
        if (index >= *m_size)
        {
            throw std::runtime_error("Attempted to access data out of bounds.");
        }
        return get()[index];
    }

    //! Get the size of the current array.
    unsigned int size() const
    {
        return *m_size;
    }

    //! Make a deep copy of this array (i.e. copy the underlying data).
    /*! This function returns by value since all memory ownership information
     *  is transmitted via shared pointers. 
     */
    void *deepCopy()
    {
        ManagedArray<T> *deep_copy = new ManagedArray<T>(size());
        // This *m_data will be a different shared_ptr than the current
        // instance's, so assigning to it directly shouldn't affect any other
        // arrays pointing to the memory of the current one and reallocating
        // the memory of others won't affect this one.
        *((*deep_copy).m_data) = *m_data;
        return deep_copy;
    }
        
private:
    //! Reallocate the data array into a new chunk of memory.
    void create_new_array(unsigned int size)
    {
        // Always allocate at least size 1
        unsigned int new_size = size > 1 ? size : 1;
        *m_data = std::shared_ptr<T>(new T[new_size], std::default_delete<T[]>());
    }
        
    std::shared_ptr<std::shared_ptr<T> > m_data;                 //!< Pointer to array.
    std::shared_ptr<unsigned int> m_size;      //!< Size of array.
};

}; }; // end namespace freud::util

#endif
