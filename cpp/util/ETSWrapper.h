#ifndef ETSWRAPPER_H
#define ETSWRAPPER_H

#include <tbb/tbb.h>

namespace freud { namespace util {

//! Wrapper classfor enumerable_thread_specific<T*>
/*! It is expected that default value for T is 0.
 */
template<typename T> class ThreadStorage
{
public:
    //! Default constructor
    ThreadStorage() : m_size(0), array(tbb::enumerable_thread_specific<T*>([]() { return nullptr; })) {}

    //! Constructor with specific size for thread local arrays
    /*! \param size Size of the thread local arrays
     */
    ThreadStorage(unsigned int size)
        : m_size(size),
          array(tbb::enumerable_thread_specific<T*>([this]() { return makeNewEmptyArray(m_size); }))
    {}

    //! Destructor
    ~ThreadStorage()
    {
        deleteArray();
    }

    //! Update size of the thread local arrays
    /*! \param size New size of the thread local arrays
     */
    void resize(unsigned int size)
    {
        if (size != m_size)
        {
            m_size = size;
            deleteArray();
            array = tbb::enumerable_thread_specific<T*>([this]() { return makeNewEmptyArray(m_size); });
        }
    }

    //! Reset the contents of thread local arrays to be 0
    void reset()
    {
        for (auto i = array.begin(); i != array.end(); ++i)
        {
            memset((void*) (*i), 0, sizeof(T) * m_size);
        }
    }

    typename tbb::enumerable_thread_specific<T*>::const_iterator begin() const
    {
        return array.begin();
    }

    typename tbb::enumerable_thread_specific<T*>::iterator begin()
    {
        return array.begin();
    }

    typename tbb::enumerable_thread_specific<T*>::const_iterator end() const
    {
        return array.end();
    }

    typename tbb::enumerable_thread_specific<T*>::iterator end()
    {
        return array.end();
    }

    typename tbb::enumerable_thread_specific<T*>::reference local()
    {
        return array.local();
    }

private:
    //! Delete arrays
    void deleteArray()
    {
        for (auto i = array.begin(); i != array.end(); ++i)
        {
            delete[](*i);
            *i = nullptr;
        }
    }

    //! Make new empty array
    /*! \param size Size of the thread local arrays
     */
    T* makeNewEmptyArray(unsigned int size)
    {
        T* tmp = new T[size];
        memset((void*) tmp, 0, sizeof(T) * size);
        return tmp;
    }

    unsigned int m_size; //!< size of thread local array
    tbb::enumerable_thread_specific<T*> array; //!< public to expose all functions
};


}; }; // end namespace freud::util

#endif
