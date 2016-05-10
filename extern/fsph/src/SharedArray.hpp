// SharedArray.hpp
// by Matthew Spellings <mspells@umich.edu>

#ifndef __SHAREDARRAY_HPP_
#define __SHAREDARRAY_HPP_

#include <algorithm>

namespace fsph{

template<typename T> class SharedArray;

/// Shim for the SharedArray class. Wraps the reference counting and
/// pointer storage for a SharedArray.
template<typename T>
class SharedArrayShim
{
    friend class SharedArray<T>;
public:
    /// Constructor. Takes ownership of a target pointer and remembers
    /// the given length (in numbers of objects)
    SharedArrayShim(T *target, size_t length):
        m_target(target),
        m_length(length),
        m_count(1)
    {}

    /// Increase the reference count for the stored pointer
    void increment()
    {
        ++m_count;
    }

    /// Decrease the reference count for the stored pointer
    void decrement()
    {
        --m_count;
        if(!m_count)
        {
            m_length = 0;
            delete[] m_target;
            m_target = NULL;
        }
    }

private:
    /// Stored pointer
    T *m_target;
    /// Size, in number of elements
    size_t m_length;
    /// Number of references to this pointer
    size_t m_count;
};

/// Generic reference-counting shared array implementation for
/// arbitrary datatypes.
template<typename T>
class SharedArray
{
public:
    typedef T* iterator;

    /// Default constructor. Allocates nothing.
    SharedArray():
        m_shim(NULL)
    {}

    /// Target constructor: allocates a new SharedArrayShim for the
    /// given pointer and takes ownership of it.
    SharedArray(T *target, size_t length):
        m_shim(new SharedArrayShim<T>(target, length))
    {}

    /// Copy constructor: make this object point to the same array as
    /// rhs, increasing the reference count if necessary
    SharedArray(const SharedArray<T> &rhs):
        m_shim(rhs.m_shim)
    {
        if(m_shim)
            m_shim->increment();
    }

    /// Destructor: decrement the reference count and deallocate if we
    /// are the last owner of the pointer
    ~SharedArray()
    {
        release();
    }

    /// Non-operator form of assignment
    void copy(const SharedArray<T> &rhs)
    {
        *this = rhs;
    }

    /// Returns true if m_shim is null or m_shim's target is null
    bool isNull()
    {
        return m_shim == NULL || m_shim->m_target == NULL;
    }

    /// Assignment operator: make this object point to the same thing
    /// as rhs (and deallocate our old memory if necessary)
    void operator=(const SharedArray<T> &rhs)
    {
        if(this != &rhs)
        {
            SharedArray<T> cpy(rhs);
            swap(cpy);
        }
    }

    /// Returns a standard style iterator to the start of the array
    iterator begin()
    {
        return get();
    }

    /// Returns a standard style iterator to just past the end of the array
    iterator end()
    {
        return get() + size();
    }

    /// Returns the raw pointer held (NULL otherwise)
    T *get()
    {
        if(m_shim)
            return m_shim->m_target;
        else
            return NULL;
    }

    /// Returns the size, in number of objects, of this array
    size_t size() const
    {
        if(m_shim)
            return m_shim->m_length;
        else
            return 0;
    }

    /// Release our claim on the pointer, including decrementing the
    /// reference count
    void release()
    {
        if(m_shim)
        {
            m_shim->decrement();
            if(m_shim->m_target == NULL)
                delete m_shim;
        }
        m_shim = NULL;
    }

    /// Stop managing this array and give it to C.
    T *disown()
    {
        T *result(NULL);
        if(m_shim)
        {
            result = m_shim->m_target;
            delete m_shim;
            m_shim = NULL;
        }
        return result;
    }

    /// Swap the contents of this array with another
    void swap(SharedArray<T> &target)
    {
        std::swap(m_shim, target.m_shim);
    }

    /// Access elements by index
    T &operator[](size_t idx)
    {
        return m_shim->m_target[idx];
    }

    /// Const access to elements by index
    const T &operator[](size_t idx) const
    {
        return m_shim->m_target[idx];
    }

private:
    /// Our pointer to the shim, which holds the array pointer and
    /// reference count
    SharedArrayShim<T> *m_shim;
};

}

#endif
