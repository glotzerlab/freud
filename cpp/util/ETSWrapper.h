#ifndef ETSWRAPPER_H
#define ETSWRAPPER_H

#include <tbb/tbb.h>

namespace freud { namespace util {

//! Wrapper classfor enumerable_thread_specific<T*>
/*! It is expected that default value for T is 0.
 */
template<typename T> class ETSArrayWrapper
{
public:
    tbb::enumerable_thread_specific<T*> array; //!< public to expose all functions

    //! Default constructor
    ETSArrayWrapper() : m_size(0), array(tbb::enumerable_thread_specific<T*>([]() { return nullptr; })) {}

    //! Constructor with specific size for thread local arrays
    /*! \param size Size of the thread local arrays
     */
    ETSArrayWrapper(unsigned int size)
        : m_size(size),
          array(tbb::enumerable_thread_specific<T*>([this]() { return makeNewEmptyArray(m_size); }))
    {}

    //! Destructor
    ~ETSArrayWrapper()
    {
        deleteArray();
    }

    //! Update size of the thread local arrays
    /*! \param size New size of the thread local arrays
     */
    void updateSize(unsigned int size)
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
};


}; }; // end namespace freud::util

#endif
