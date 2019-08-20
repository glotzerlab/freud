#ifndef NUMERICAL_ARRAY_H
#define NUMERICAL_ARRAY_H

#include <memory>
#include <vector>

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
 *  of freud. To support multidimensional arrays, the underlying data is stored
 *  in a linear array that can be indexed into according to standard freud
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
 */
template<typename T> class ManagedArray
{
public:
    //! Constructor based on a shape tuple.
    /*! Including a default value for the shape allows the usage of this
     *  constructor as the default constructor.
     *  \param shape Shape of the array to allocate.
     */
    ManagedArray(std::vector<unsigned int> shape = {0})
    {
        m_shape = std::make_shared<std::vector<unsigned int> >(shape);

        m_data = std::shared_ptr<std::shared_ptr<T> >(
            new std::shared_ptr<T>(new T[size()], std::default_delete<T[]>()));
        reset();
    }

    //! Destructor (currently empty because data is managed by shared pointer).
    ~ManagedArray() {}

    //! Update size of the array.
    /*! \param size New size of the array.
     */
    void resize(unsigned int size)
    {
        resize(std::vector<unsigned int> {size});
    }

    //! Update size of the array.
    /*! \param shape New shape of the array.
     */
    void resize(std::vector<unsigned int> shape)
    {
        if (shape != *m_shape)
        {
            *m_shape = shape;
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
        *m_data = std::shared_ptr<T>(new T[size()], std::default_delete<T[]>());
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
            throw std::runtime_error("Attempted to access data out of bounds.");
        }
        return get()[index];
    }

    //! Read-only index into array.
    const T &operator[](unsigned int index) const
    {
        if (index >= size())
        {
            throw std::runtime_error("Attempted to access data out of bounds.");
        }
        return get()[index];
    }

    //! Get the size of the current array.
    unsigned int size() const
    {
        unsigned int size = 1;
        for (unsigned int i = 0; i < m_shape->size(); i++)
        {
            size *= (*m_shape)[i];
        }
        return size;
    }

    //! Get the shape of the current array.
    std::vector<unsigned int> shape() const
    {
        return *m_shape;
    }

    //! Dissociate this ManagedArray from others referencing the same data.
    /*! ManagedArrays share ownership of an array of data using a pointer to a
     * pointer to the data, such that all arrays sharing data have distinct
     * top-level pointers all pointing to the same second pointer. If we want
     * to break this association between the current ManagedArray and the
     * others without destroying or modifying the underlying data, we need to
     * allocate a new second-level pointer for this ManagedArray and point it
     * at the same underlying data.
     */
    void dissociate()
    {
        m_data = std::shared_ptr<std::shared_ptr<T> >(
            new std::shared_ptr<T>(*m_data));
    }
        
private:
    std::shared_ptr<std::shared_ptr<T> > m_data;           //!< Pointer to array.
    std::shared_ptr<std::vector<unsigned int> > m_shape;                  //!< Size of array.
};

}; }; // end namespace freud::util

#endif
