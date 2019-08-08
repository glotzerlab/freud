#ifndef NUMERICAL_ARRAY_H
#define NUMERICAL_ARRAY_H

#include <memory>
#include <tbb/tbb.h>
#include <iostream>

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
    //! Default constructor.
    /*! For safety, these arrays must be constructed with the manage argument
     *  set to true in order to create an empty instance that manages its own
     *  memory. By default, constructed arrays do not manage their own memory to
     *  avoid accidental memory leaks.
     *
     *  \param manage Whether or not this instance should manage its own memory
     *                (Default: false).
     */
    ManagedArray(bool manage=false) : m_size(nullptr), m_data(nullptr), m_managed(manage), m_external_ptr(false) {}

    //! Constructor with specific size for thread local arrays
    /*! When using this constructor, the class automatically manages its own
     *  memory since it is allocating it.
     *
     *  \param size Size of the array to allocate.
     */
    ManagedArray(unsigned int size) : m_managed(true), m_external_ptr(false)
    {
        m_size = new unsigned int(size);
        m_data = new T[size];
        reset();
    }

    //! Copy constructor.
    /*! This constructor is required to ensure that the original object always
     *  owns its own memory. However ManagedArrays pointing to external data will always store
     *
     *  \param other ManagedArray instance to copy.
     */
    ManagedArray(const ManagedArray &other) : m_data(other.m_data), m_managed(false), m_external_ptr(other.m_external_ptr)
    {
        if (m_external_ptr)
        {
            m_size = new unsigned int(*other.m_size);
        }
        else
        {
            m_size = other.m_size;
        }
    }

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

        m_data = other.m_data;
        m_external_ptr = other.m_external_ptr;
        if (m_external_ptr)
        {
            m_size = new unsigned int(*other.m_size);
        }
        else
        {
            m_size = other.m_size;
        }
        return *this;
    }

    //! Destructor (currently empty because data is managed by shared pointer).
    ~ManagedArray()
    {
        if (m_managed)
            delete[] m_data;
        if (m_managed || m_external_ptr)
            delete m_size;
    }

    //! Copy another ManagedArray and obtain ownership of its data.
    /*! This method allows two ManagedArray instances to trade ownership
     *  characteristics. The semantics only make sense if one array is managing
     *  its own memory and the other is not. For conceptual simplicity, an array
     *  that is initialized as pointing to external data cannot acquire data,
     *  but in principle there is no reason to prevent this. However, that usage
     *  pattern is error-prone, so preventing helps avoid subtle memory bugs.
     *
     *  The primary purpose of this function is to support more natural
     *  behavior of the Python API by taking advantage of Python reference
     *  counting to avoid bad array references. In particular, judicious use of
     *  this method prevents numpy arrays from becoming outdated. The expected
     *  usage is that Cython mirrors of C++ compute classes will take ownership
     *  of data once it has been computed, and only return the ownership to the
     *  C++ class if no existing numpy arrays are referencing the array in
     *  Python.
     *
     *  \param other ManagedArray instance to copy.
     */
    void acquire(ManagedArray &other)
    {
        if (!other.m_managed)
        {
            throw std::runtime_error("Can only acquire data from a ManagedArray that owns its own data.");
        } else if (m_managed || m_external_ptr)
        {
            throw std::runtime_error("A ManagedArray that owns data cannot acquire another's data.");
        }

        m_data = other.m_data;
        m_size = other.m_size;
        m_managed = true;
        other.m_managed = false;
    }

    //! Reallocate memory for this array.
    /*! This method may only be called for arrays not currently managing their
     *  own memory (to avoid memory leaks). It allocates new memory for the
     *  array.
     */
    void reallocate()
    {
        if (m_managed)
        {
            throw std::runtime_error("You cannot reallocate a ManagedArray that is currently managing its own memory.");
        } else if (m_external_ptr)
        {
            throw std::runtime_error("Reallocation is not allowed for ManagedArrays pointing to external data.");
        }

        if (m_size != nullptr)
        {
            m_size = new unsigned int(size());
            m_data = new T[*m_size];
        }

        m_managed = true;
        m_external_ptr = false;
        reset();
    }

    //! Update size of the array.
    /*! \param size New size of the array.
     */
    void resize(unsigned int size)
    {
        if (!m_managed)
        {
            throw std::runtime_error("ManagedArray can only resize arrays it is managing.");
        }
        if (size != this->size())
        {
            if (this->size())
                delete[](m_data);

            if (size != 0)
            {
                if (m_size == nullptr)
                {
                    m_size = new unsigned int(size);
                }
                else
                {
                    *m_size = size;
                }
            }
            else
            {
                m_size = nullptr;
            }

            if (this->size())
                m_data = new T[size];
            reset();
        }
    }

    //! Reset the contents of array to be 0.
    void reset()
    {
        if (!m_managed)
        {
            throw std::runtime_error("ManagedArray can only reset arrays it is managing.");
        }

        if (size())
            memset((void*) m_data, 0, sizeof(T) * size());
    }

    //! Return the underlying pointer.
    T *get() const
    {
        return m_data;
    }

    //! Writeable index into array.
    T &operator[](unsigned int index)
    {
        if (index >= size())
        {
            throw std::runtime_error("Attempted to access data out of bounds.");
        }
        return m_data[index];
    }

    //! Read-only index into array.
    const T &operator[](unsigned int index) const
    {
        if (index >= size())
        {
            throw std::runtime_error("Attempted to access data out of bounds.");
        }
        return m_data[index];
    }

    //! Get the size of the current array.
    unsigned int size() const
    {
        if (m_size == nullptr)
            return 0;
        return *m_size;
    }

    //! Check if the array manages its own memory.
    bool isManaged() const
    {
        return m_managed;
    }

private:
    unsigned int *m_size;        //!< Size of array.
    T *m_data;  //!< Pointer to array.
    bool m_managed;  //!< Whether or not the array should be managing its own data.
    bool m_external_ptr;  //!< Whether or not the array points to another (non-ManagedArray) data source (set upon initialization or copy).
};

}; }; // end namespace freud::util

#endif
