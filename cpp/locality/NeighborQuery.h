// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef NEIGHBOR_QUERY_H
#define NEIGHBOR_QUERY_H

#include <memory>
#include <stdexcept>
#include <tbb/tbb.h>

#include "Box.h"
#include "NeighborList.h"

/*! \file NeighborQuery.h
    \brief Defines the abstract API for collections of points that can be
           queried against for neighbors.
*/

namespace freud { namespace locality {

//! Simple data structure encoding neighboring points.
/*! The primary purpose of this class is to provide a more meaningful struct
 *  than a simple std::pair, which is hard to interpret. Additionally, this
 *  class defines the less than operator according to distance, making it
 *  possible to sort.
 */
struct NeighborPoint
{
    NeighborPoint() : id(0), ref_id(0), distance(0) {}

    NeighborPoint(unsigned int id, unsigned int ref_id, float d) : id(id), ref_id(ref_id), distance(d) {}

    //! Equality checks both id and distance.
    bool operator==(const NeighborPoint& n)
    {
        return (id == n.id) && (ref_id == n.ref_id) && (distance == n.distance);
    }

    //! Default comparator of points is by distance.
    /*! This form of comparison allows easy sorting of nearest neighbors by
     *  distance.
     */
    bool operator<(const NeighborPoint& n) const
    {
        return distance < n.distance;
    }

    unsigned int id;     //! The point id.
    unsigned int ref_id; //! The reference point id.
    float distance;      //! The distance between the point and the reference point.
};

//! (Almost) POD class to hold information about generic queries.
/*! This class provides a standard method for specifying the type of query to
 *  perform with a NeighborQuery object. Rather than calling queryBall
 *  specifically, for example, the user can call a generic querying function and
 *  provide an instance of this class to specify the nature of the query.
 */
struct QueryArgs
{
    //! Define constructor
    /*! We must violate the strict POD nature of the class to support default
     *  values for parameters.
     */
    QueryArgs() : nn(-1), rmax(-1), scale(-1), exclude_ii(false) {}

    enum QueryType
    {
        ball,   //! Query based on distance cutoff.
        nearest //! Query based on number of requested neighbors.
    };

    QueryType mode; //! Whether to perform a ball or k-nearest neighbor query.
    int nn;         //! The number of nearest neighbors to find.
    float rmax;     //! The cutoff distance within which to find neighbors
    float scale; //! The scale factor to use when performing repeated ball queries to find a specified number
                 //! of nearest neighbors.
    bool exclude_ii; //! If true, exclude self-neighbors.
};

// Forward declare the iterator
class NeighborQueryIterator;

//! Parent data structure for all neighbor finding algorithms.
/*! This class defines the API for all data structures for accelerating
 *  neighbor finding. The object encapsulates a set of points and a system box
 *  that define the set of points to search and the periodic system within these
 *  points can be found.
 *
 *  The primary interface to this class is through the query and queryBall
 *  methods, which support k-nearest neighbor queries and distance-based
 *  queries, respectively.
 */
class NeighborQuery
{
public:
    //! Nullary constructor for Cython
    NeighborQuery() {}

    //! Constructor
    NeighborQuery(const box::Box& box, const vec3<float>* ref_points, unsigned int Nref)
        : m_box(box), m_ref_points(ref_points), m_Nref(Nref)
    {}

    //! Empty Destructor
    virtual ~NeighborQuery() {}

    //! Perform a query based on a set of query parameters.
    /*! Given a QueryArgs object and a set of points to perform a query
     *  with, this function will dispatch the query to the appropriate
     *  querying function.
     *
     *  This function should just be called query, but Cython's function
     *  overloading abilities seem buggy at best, so it's easiest to just
     *  rename the function.
     */
    virtual std::shared_ptr<NeighborQueryIterator> queryWithArgs(const vec3<float>* points, unsigned int N,
                                                                 QueryArgs args)
    {
        this->validateQueryArgs(args);
        if (args.mode == QueryArgs::ball)
        {
            return this->queryBall(points, N, args.rmax, args.exclude_ii);
        }
        else if (args.mode == QueryArgs::nearest)
        {
            return this->query(points, N, args.nn, args.exclude_ii);
        }
        else
        {
            throw std::runtime_error("Invalid query mode provided to generic query function.");
        }
    }

    //! Given a point, find the k elements of this data structure
    //  that are the nearest neighbors for each point.
    virtual std::shared_ptr<NeighborQueryIterator> query(const vec3<float>* points, unsigned int N,
                                                         unsigned int k, bool exclude_ii = false) const = 0;

    //! Given a point, find all elements of this data structure
    //  that are within a certain distance r.
    virtual std::shared_ptr<NeighborQueryIterator> queryBall(const vec3<float>* points, unsigned int N,
                                                             float r, bool exclude_ii = false) const = 0;

    //! Get the simulation box
    const box::Box& getBox() const
    {
        return m_box;
    }

    //! Get the reference points
    const vec3<float>* getRefPoints() const
    {
        return m_ref_points;
    }

    //! Get the number of reference points
    const unsigned int getNRef() const
    {
        return m_Nref;
    }

    //! Get a point's coordinates using index operator notation
    const vec3<float> operator[](unsigned int index) const
    {
        if (index >= m_Nref)
        {
            throw std::runtime_error("NeighborQuery attempted to access a point with index >= Nref.");
        }
        return m_ref_points[index];
    }

protected:
    virtual void validateQueryArgs(QueryArgs& args)
    {
        if (args.mode == QueryArgs::ball)
        {
            if (args.rmax == -1)
                throw std::runtime_error("You must set rmax in the query arguments.");
        }
        else if (args.mode == QueryArgs::nearest)
        {
            if (args.nn == -1)
                throw std::runtime_error("You must set nn in the query arguments.");
        }
    }

    const box::Box m_box;            //!< Simulation box where the particles belong
    const vec3<float>* m_ref_points; //!< Reference point coordinates
    unsigned int m_Nref;             //!< Number of reference points
};

//! The iterator class for neighbor queries on NeighborQuery objects.
/*! This is an abstract class that defines the abstract API for neighbor
 *  iteration. All subclasses of NeighborQuery should also subclass
 *  NeighborQueryIterator and define the next() method appropriately. The next()
 *  method is the primary mode of interaction with the iterator, and allows
 *  looping through the iterator.
 *
 *  Note that due to the fact that there is no way to know when iteration is
 *  complete until all relevant points are actually checked (irrespective of the
 *  underlying data structure), the end() method will not return true until the
 *  next method reaches the end of control flow at least once without finding a
 *  next neighbor. As a result, the next() method is required to return
 *  NeighborQueryIterator::ITERATOR_TERMINATOR on all calls after the last neighbor is
 *  found in order to guarantee that the correct set of neighbors is considered.
 */
class NeighborQueryIterator
{
public:
    //! Nullary constructor for Cython
    NeighborQueryIterator() {}

    //! Constructor
    NeighborQueryIterator(const NeighborQuery* neighbor_query, const vec3<float>* points, unsigned int N,
                          bool exclude_ii)
        : m_neighbor_query(neighbor_query), m_points(points), m_N(N), cur_p(0), m_finished(false),
          m_exclude_ii(exclude_ii)
    {}

    //! Empty Destructor
    virtual ~NeighborQueryIterator() {}

    //! Indicate when done.
    virtual bool end()
    {
        return m_finished;
    }

    //! Replicate this class's query on a per-particle basis.
    /*! Note that because this query is on a per-particle basis, there is
     *  no reason to support ii exclusion, so we neglect that here.
     */
    virtual std::shared_ptr<NeighborQueryIterator> query(unsigned int idx)
    {
        throw std::runtime_error("The query method must be implemented by child classes.");
    }

    //! Get the next element.
    virtual NeighborPoint next()
    {
        throw std::runtime_error("The next method must be implemented by child classes.");
    }

    //! Generate a NeighborList from query.
    /*! This function exploits parallelism by finding the neighbors for
     *  each query point in parallel and adding them to a list, which is
     *  then sorted in parallel as well before being added to the
     *  NeighborList object. Right now this won't be backwards compatible
     *  because the kn query is not symmetric, so even if we reverse the
     *  output order here the actual neighbors found will be different.
     *
     *  This function returns a pointer, not a shared pointer, so the
     *  caller is responsible for deleting it. The reason for this is that
     *  the primary use-case is to have this object be managed by instances
     *  of the Cython NeighborList class.
     */
    virtual NeighborList* toNeighborList()
    {
        typedef tbb::enumerable_thread_specific<std::vector<std::pair<size_t, size_t>>> BondVector;
        BondVector bonds;
        tbb::parallel_for(tbb::blocked_range<size_t>(0, m_N), [&](const tbb::blocked_range<size_t>& r) {
            BondVector::reference local_bonds(bonds.local());
            NeighborPoint np;
            for (size_t i(r.begin()); i != r.end(); ++i)
            {
                std::shared_ptr<NeighborQueryIterator> it = this->query(i);
                while (!it->end())
                {
                    np = it->next();
                    // If we're excluding ii bonds, we have to check before adding.
                    if (!m_exclude_ii || i != np.ref_id)
                    {
                        // Swap ref_id and id order for backwards compatibility.
                        local_bonds.emplace_back(np.ref_id, i);
                    }
                }
                // Remove the last item, which is just the terminal sentinel value.
                local_bonds.pop_back();
            }
        });

        tbb::flattened2d<BondVector> flat_bonds = tbb::flatten2d(bonds);
        std::vector<std::pair<size_t, size_t>> linear_bonds(flat_bonds.begin(), flat_bonds.end());
        tbb::parallel_sort(linear_bonds.begin(), linear_bonds.end());

        unsigned int num_bonds = linear_bonds.size();

        NeighborList* nl = new NeighborList();
        nl->resize(num_bonds);
        nl->setNumBonds(num_bonds, m_neighbor_query->getNRef(), m_N);
        size_t* neighbor_array(nl->getNeighbors());
        float* neighbor_weights(nl->getWeights());

        parallel_for(tbb::blocked_range<size_t>(0, num_bonds), [&](const tbb::blocked_range<size_t>& r) {
            for (size_t bond(r.begin()); bond < r.end(); ++bond)
            {
                neighbor_array[2 * bond] = linear_bonds[bond].first;
                neighbor_array[2 * bond + 1] = linear_bonds[bond].second;
            }
        });
        memset((void*) neighbor_weights, 1, sizeof(float) * linear_bonds.size());

        return nl;
    }

    static const NeighborPoint ITERATOR_TERMINATOR; //!< The object returned when iteration is complete.

protected:
    const NeighborQuery* m_neighbor_query; //!< Link to the NeighborQuery object.
    const vec3<float>* m_points;           //!< Coordinates of query points.
    unsigned int m_N;                      //!< Number of points.
    unsigned int cur_p;                    //!< The current index into the points (bounded by m_N).

    unsigned int
        m_finished;    //!< Flag to indicate that iteration is complete (must be set by next on termination).
    bool m_exclude_ii; //!< Flag to indicate whether or not to include self bonds.
};

//! Iterator for nearest neighbor queries.
/*! The primary purpose for this class is to ensure that conversion of
 *  k-nearest neighbor queries into NeighborList objects correctly handles
 *  self-neighbor exclusions. This problem arises because the generic
 *  toNeighborList function does not pass the exclude_ii argument through to
 *  the calls to query that generate new query iterators, but rather filters
 *  out these exclusions after the fact. The reason it does this is because it
 *  performs queries on a per-particle basis, so the indices cannot match.
 *  However, this leads to a new problem, which is that when self-neighbors are
 *  excluded, one fewer neighbor is found than desired. This class overrides
 *  that behavior to ensure that the correct number of neighbors is found.
 */
class NeighborQueryQueryIterator : virtual public NeighborQueryIterator
{
public:
    //! Constructor
    NeighborQueryQueryIterator(const NeighborQuery* neighbor_query, const vec3<float>* points, unsigned int N,
                               bool exclude_ii, unsigned int k)
        : NeighborQueryIterator(neighbor_query, points, N, exclude_ii), m_count(0), m_k(k),
          m_current_neighbors()
    {}

    //! Empty Destructor
    virtual ~NeighborQueryQueryIterator() {}

    //! Generate a NeighborList from query.
    /*! This function is a thin wrapper around the parent class function.
     * All it needs to do is increase the counter of points to find, find
     * them, and then reset.
     */
    virtual NeighborList* toNeighborList()
    {
        NeighborList* nlist;
        if (m_exclude_ii)
            m_k += 1;
        try
        {
            nlist = NeighborQueryIterator::toNeighborList();
        }
        catch (...)
        {
            if (m_exclude_ii)
                m_k -= 1;
            throw;
        }
        return nlist;
    }

protected:
    unsigned int m_count;                           //!< Number of neighbors returned for the current point.
    unsigned int m_k;                               //!< Number of nearest neighbors to find
    std::vector<NeighborPoint> m_current_neighbors; //!< The current set of found neighbors.
};
}; }; // end namespace freud::locality

#endif // NEIGHBOR_QUERY_H
