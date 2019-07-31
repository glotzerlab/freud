#ifndef NUMERICAL_ARRAY_H
#define NUMERICAL_ARRAY_H

#include <memory>
#include <tbb/tbb.h>

/*! \file NumericalArray.h
    \brief Defines the standard array class to be used throughout freud.
*/

namespace freud { namespace util {

//! RAII class to handle the storage of all arrays of numerical data used in freud.
/*! The purpose of this class is to handle standard memory management, and to
 * provide an abstraction around the implementation-specific choice of
 * underlying data structure for arrays of data in freud. We specify that it
 * only encompasses numerical data to allow e.g. memset to 0 as a valid
 * operation.
 */
template<typename T> class NumericalArray
{
public:
    //! Default constructor
    NumericalArray() : m_size(0) {}

    //! Constructor with specific size for thread local arrays
    /*! \param size Size of the array to allocate.
     */
    NumericalArray(unsigned int size) : m_size(size)
    {
        m_data = std::shared_ptr<T>(new T[size], std::default_delete<T[]>());
        reset();
    }

    //! Destructor (currently empty because data is managed by shared pointer).
    ~NumericalArray() {}

    //! Update size of the thread local arrays.
    /*! \param size New size of the thread local arrays.
     */
    void resize(unsigned int size)
    {
        if (size != m_size)
        {
            m_size = size;
            m_data = std::shared_ptr<T>(new T[size], std::default_delete<T[]>());
            reset();
        }
    }

    //! Reset the contents of thread local arrays to be 0.
    void reset()
    {
        memset((void*) m_data.get(), 0, sizeof(T) * m_size);
    }

    //! Return the underlying pointer.
    std::shared_ptr<T> getData() const
    {
        return m_data;
    }

    //! Writeable index into array.
    T &operator[](unsigned int index)
    {
        if (index >= m_size)
        {
            throw std::runtime_error("Attempted to access data out of bounds.");
        }
        return m_data.get()[index];
    }

private:
    unsigned int m_size;        //!< Size of array.
    std::shared_ptr<T> m_data;  //!< Pointer to array.
};

}; }; // end namespace freud::util

#endif
