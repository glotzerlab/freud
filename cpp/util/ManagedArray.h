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
 * provide an abstraction around the implementation-specific choice of
 * underlying data structure for arrays of data in freud. We specify that it
 * only encompasses numerical data to allow e.g. memset to 0 as a valid
 * operation.
 */
template<typename T> class ManagedArray
{
public:
    //! Default constructor.
    /*! For safety, these arrays must be constructed with the manage argument
     *  set to true in order to create an empty instance that manages its own
     *  memory. By default, constructed arrays do not manage their own memory to
     *  avoid accidental memory leaks.
     *
     *  \param manage Whether or not this instance should manage its own memory
     *                (Default: false).
     */
    ManagedArray(bool manage=false) : m_size(0), m_data(nullptr), m_managed(manage) {}

    //! Constructor with specific size for thread local arrays
    /*! When using this constructor, the class automatically manages its own
     *  memory since it is allocating it.
     *
     *  \param size Size of the array to allocate.
     */  
    ManagedArray(unsigned int size) : m_size(size), m_managed(true)
    {
        m_data = new T[size];
        reset();
    }

    //! Construct object from existing array.
    /*! \param T* Pointer to existing data.
     *  \param size Size of the array to allocate.
     */
    ManagedArray(T* array, unsigned int size) : m_size(size), m_managed(false)
    {
        m_data = array;
    }

    //! Copy constructor.
    /*! This constructor is required to ensure that the original object always
     *  owns its own memory.
     *
     *  \param other ManagedArray instance to copy.
     */  
    ManagedArray(const ManagedArray &other) : m_size(other.size()), m_data(other.get()), m_managed(false) {}

    //! Copy assignment.
    /*! Similar to the copy constructor, this operator must be defined to
     *  ensure that the original object always owns its own memory.
     *
     *  \param first ManagedArray instance to copy.
     */  
    ManagedArray& operator= (const ManagedArray &other) 
    { 
        if (m_managed)
        {
            throw std::runtime_error("You cannot assign to a ManagedArray that is currently managing its own memory.");
        }

        m_size = other.size();
        m_data = other.get();
        m_managed = false;
        return *this; 
    }

    //! Destructor (currently empty because data is managed by shared pointer).
    ~ManagedArray()
    {
        if (m_managed)
            delete[](m_data);
    }

    //! Factory function to copy a ManagedArray and obtain ownership of its data.
    /*! The purpose of this function is to explicitly construct a copy that
     *  will now be in charge of managing the data. Separating this logic into a
     *  factory function forces the user to think very explicitly about which
     *  instance is responsible for data management to avoid any issues with
     *  memory leaks or multiple frees. The primary use-case is in the Python
     *  API for compute classes so that the Cython mirror classes can acquire
     *  ownership of exposed arrays after they have been computed.
     *
     *  \param first ManagedArray instance to copy.
     */
    static ManagedArray *createAndAcquire(ManagedArray &other)
    {
        if (!other.m_managed)
        {
            throw std::runtime_error("Can only acquire data from a ManagedArray that owns its own data.");
        }

        ManagedArray *new_arr = new ManagedArray();
        new_arr->m_data = other.get();
        new_arr->m_size = other.size();
        new_arr->m_managed = true;
        other.m_managed = false;

        return new_arr;
    }

    //! Reallocate memory for this array.
    /*! This method may only be called for arrays not currently managing their
     *  own memory (to avoid memory leaks. It allocates new memory for the
     *  array.
     */
    void reallocate()
    {
        if (m_managed)
        {
            throw std::runtime_error("You cannot reallocate a ManagedArray that is currently managing its own memory.");
        }
        m_data = new T[m_size];
        m_managed = true;
        reset();
    }

    //! Update size of the thread local arrays.
    /*! \param size New size of the thread local arrays.
     */
    void resize(unsigned int size)
    {
        if (!m_managed)
        {
            throw std::runtime_error("ManagedArray can only resize arrays it is managing.");
        }
        if (size != m_size)
        {
            if (m_size > 0)
                delete[](m_data);
            m_size = size;
            m_data = new T[size];
            reset();
        }
    }

    //! Reset the contents of thread local arrays to be 0.
    void reset()
    {
        if (!m_managed)
        {
            throw std::runtime_error("ManagedArray can only reset arrays it is managing.");
        }

        memset((void*) m_data, 0, sizeof(T) * m_size);
    }

    //! Return the underlying pointer.
    T *get() const
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
        return m_data[index];
    }

    //! Read-only index into array.
    const T &operator[](unsigned int index) const
    {
        if (index >= m_size)
        {
            throw std::runtime_error("Attempted to access data out of bounds.");
        }
        return m_data[index];
    }

    //! Get the size of the current array.
    unsigned int size() const
    {
        return m_size;
    }

    //! Check if the array manages its own memory.
    bool isManaged() const
    {
        return m_managed;
    }

private:
    unsigned int m_size;        //!< Size of array.
    T *m_data;  //!< Pointer to array.
    bool m_managed;  //!< Whether or not the array should be managing its own data.
};

}; }; // end namespace freud::util

#endif
