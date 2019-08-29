// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef NEIGHBOR_QUERY_H
#define NEIGHBOR_QUERY_H

#include <memory>
#include <stdexcept>
#include <tbb/tbb.h>
#include <tuple>

#include "Box.h"
#include "NeighborBond.h"
#include "NeighborList.h"

/*! \file NeighborQuery.h
    \brief Defines the abstract API for collections of points that can be
           queried against for neighbors.
*/

namespace freud { namespace locality {

//! POD class to hold information about generic queries.
/*! This class provides a standard method for specifying the type of query to
 *  perform with a NeighborQuery object. Rather than calling queryBall
 *  specifically, for example, the user can call a generic querying function and
 *  provide an instance of this class to specify the nature of the query.
 */
struct QueryArgs
{
    //! Default constructor.
    /*! We set default values for all parameters here.
     */
    QueryArgs() : mode(DEFAULT_MODE), num_neighbors(DEFAULT_NUM_NEIGHBORS), r_max(DEFAULT_R_MAX),
                  scale(DEFAULT_SCALE), exclude_ii(DEFAULT_EXCLUDE_II) {}

    //! Enumeration for types of queries.
    enum QueryType
    {
        none,    //! Default query type to avoid implicit default types.
        ball,    //! Query based on distance cutoff.
        nearest, //! Query based on number of requested neighbors.
    };

    QueryType mode; //! Whether to perform a ball or k-nearest neighbor query.
    unsigned int num_neighbors;         //! The number of nearest neighbors to find.
    float r_max;     //! The cutoff distance within which to find neighbors
    float scale; //! The scale factor to use when performing repeated ball queries to find a specified number
                 //! of nearest neighbors.
    bool exclude_ii; //! If true, exclude self-neighbors.

    static const QueryType DEFAULT_MODE;                //!< Default mode.
    static const unsigned int DEFAULT_NUM_NEIGHBORS;        //!< Default number of neighbors.
    static const float DEFAULT_R_MAX;                   //!< Default query distance.
    static const float DEFAULT_SCALE;                   //!< Default scaling parameter for AABB nearest neighbor queries.
    static const bool DEFAULT_EXCLUDE_II;               //!< Default for whether or not to include self-neighbors.
};

// Forward declare the iterators
class NeighborQueryIterator;
class NeighborQueryPerPointIterator;

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
    NeighborQuery(const box::Box& box, const vec3<float>* points, unsigned int n_points)
        : m_box(box), m_points(points), m_n_points(n_points)
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
    virtual std::shared_ptr<NeighborQueryIterator> queryWithArgs(const vec3<float>* query_points, unsigned int n_query_points,
                                                                 QueryArgs args) const
    {
        this->validateQueryArgs(args);
        if (args.mode == QueryArgs::ball)
        {
            return this->queryBall(query_points, n_query_points, args.r_max, args.exclude_ii);
        }
        else if (args.mode == QueryArgs::nearest)
        {
            return this->query(query_points, n_query_points, args.num_neighbors, args.exclude_ii);
        }
        else
        {
            throw std::runtime_error("Invalid query mode provided to generic query function.");
        }
    }

    //! Perform a query based on a set of query parameters.
    /*! Given a QueryArgs object and a set of points to perform a query
     *  with, this function will dispatch the query to the appropriate
     *  querying function.
     *
     *  This function should just be called query, but Cython's function
     *  overloading abilities seem buggy at best, so it's easiest to just
     *  rename the function.
     */
    virtual std::shared_ptr<NeighborQueryPerPointIterator> queryWithArgs(const vec3<float> query_point, unsigned int query_point_idx,
                                                                 QueryArgs args) const = 0;

    //! Given a set of points, find the num_neighbors elements of this data structure
    //  that are the nearest neighbors for each point.
    virtual std::shared_ptr<NeighborQueryIterator> query(const vec3<float>* query_points, unsigned int n_query_points,
                                                         unsigned int num_neighbors, bool exclude_ii = false) const
    {
        QueryArgs qargs;
        qargs.mode = QueryArgs::QueryType::nearest;
        qargs.num_neighbors = num_neighbors;
        qargs.exclude_ii = exclude_ii;
        return std::make_shared<NeighborQueryIterator>(this, query_points, n_query_points, qargs);
    }


    //! Given a point, find all elements of this data structure
    //  that are within a certain distance.
    virtual std::shared_ptr<NeighborQueryIterator> queryBall(const vec3<float>* query_points, unsigned int n_query_points,
                                                             float r_max, bool exclude_ii = false) const
    {
        QueryArgs qargs;
        qargs.mode = QueryArgs::QueryType::ball;
        qargs.r_max = r_max;
        qargs.exclude_ii = exclude_ii;
        return std::make_shared<NeighborQueryIterator>(this, query_points, n_query_points, qargs);
    }

    //! Get the simulation box
    const box::Box& getBox() const
    {
        return m_box;
    }

    //! Get the reference points
    const vec3<float>* getPoints() const
    {
        return m_points;
    }

    //! Get the number of reference points
    unsigned int getNPoints() const
    {
        return m_n_points;
    }

    //! Get a point's coordinates using index operator notation
    const vec3<float> operator[](unsigned int index) const
    {
        if (index >= m_n_points)
        {
            throw std::runtime_error("NeighborQuery attempted to access a point with index >= n_points.");
        }
        return m_points[index];
    }

    //! Validate the combination of specified arguments.
    /*! Before checking if the combination of parameters currently set is
     *  valid, this function first attempts to infer a mode if one is not set in
     *  order to allow the user to specify certain simple minimal argument
     *  combinations (e.g. just an r_max) without having to specify the mode
     *  explicitly.
     */
    virtual void validateQueryArgs(QueryArgs& args) const
    {
        inferMode(args);
        // Validate remaining arguments.
        if (args.mode == QueryArgs::ball)
        {
            if (args.r_max == QueryArgs::DEFAULT_R_MAX)
                throw std::runtime_error("You must set r_max in the query arguments when performing ball queries.");
            if (args.num_neighbors != QueryArgs::DEFAULT_NUM_NEIGHBORS)
                throw std::runtime_error("You cannot set num_neighbors in the query arguments when performing ball queries.");
        }
        else if (args.mode == QueryArgs::nearest)
        {
            if (args.num_neighbors == QueryArgs::DEFAULT_NUM_NEIGHBORS)
                throw std::runtime_error("You must set num_neighbors in the query arguments when performing number of neighbor queries.");
        }
    }


protected:
    //! Try to determine the query mode if one is not specified.
    /*! If no mode is specified and a number of neighbors is specified, the
     *  query mode must be a nearest neighbors query (all other arguments can
     *  reasonably modify that query). Otherwise, if a max distance is set we
     *  can assume a ball query is desired.
     */
    virtual void inferMode(QueryArgs& args) const
    {
        // Infer mode if possible.
        if (args.mode == QueryArgs::none)
        {
            if (args.num_neighbors != QueryArgs::DEFAULT_NUM_NEIGHBORS)
            {
                args.mode = QueryArgs::nearest;
            }
            else if (args.r_max != QueryArgs::DEFAULT_R_MAX)
            {
                args.mode = QueryArgs::ball;
            }
        }
    }

    const box::Box m_box;            //!< Simulation box where the particles belong
    const vec3<float>* m_points; //!< Reference point coordinates
    unsigned int m_n_points;             //!< Number of reference points
};

class NeighborPerPointIterator
{
public:
    //! Nullary constructor for Cython
    NeighborPerPointIterator() {}

    //! Constructor
    NeighborPerPointIterator(unsigned int query_point_idx)
        : m_query_point_idx(query_point_idx) {}

    //! Empty Destructor
    virtual ~NeighborPerPointIterator() {}

    //! Indicate when done.
    virtual bool end() = 0;

    //! Get the next element.
    virtual NeighborBond next() = 0;

    static const NeighborBond ITERATOR_TERMINATOR; //!< The object returned when iteration is complete.

protected:
    unsigned int m_query_point_idx; //!< The index of the query point.
};

//! The iterator class for neighbor queries on NeighborQuery objects.
/*! This is an abstract class that defines the abstract API for neighbor
 *  iteration. All subclasses of NeighborQuery should also subclass
 *  NeighborQueryPerPointIterator and define the next() method appropriately. The next()
 *  method is the primary mode of interaction with the iterator, and allows
 *  looping through the iterator.
 *
 *  Note that due to the fact that there is no way to know when iteration is
 *  complete until all relevant points are actually checked (irrespective of the
 *  underlying data structure), the end() method will not return true until the
 *  next method reaches the end of control flow at least once without finding a
 *  next neighbor. As a result, the next() method is required to return
 *  NeighborQueryPerPointIterator::ITERATOR_TERMINATOR on all calls after the last neighbor is
 *  found in order to guarantee that the correct set of neighbors is considered.
 */
class NeighborQueryPerPointIterator : public NeighborPerPointIterator
{
public:
    //! Nullary constructor for Cython
    NeighborQueryPerPointIterator() {}

    //! Constructor
    NeighborQueryPerPointIterator(const NeighborQuery* neighbor_query, const vec3<float> query_point, unsigned int query_point_idx,
                          bool exclude_ii)
        : NeighborPerPointIterator(query_point_idx), m_neighbor_query(neighbor_query), m_query_point(query_point), m_finished(false), m_exclude_ii(exclude_ii) {}

    //! Empty Destructor
    virtual ~NeighborQueryPerPointIterator() {}

    //! Indicate when done.
    virtual bool end()
    {
        return m_finished;
    }

    //! Get the next element.
    virtual NeighborBond next()
    {
        throw std::runtime_error("The next method must be implemented by child classes.");
    }

    static const NeighborBond ITERATOR_TERMINATOR; //!< The object returned when iteration is complete.

protected:
    const NeighborQuery* m_neighbor_query; //!< Link to the NeighborQuery object.
    const vec3<float> m_query_point;           //!< Coordinates of the query point.
    unsigned int cur_p;                    //!< The current index into the points (bounded by m_n_query_points).
    bool m_finished;    //!< Flag to indicate that iteration is complete (must be set by next on termination).
    bool m_exclude_ii; //!< Flag to indicate whether or not to include self bonds.
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
    NeighborQueryIterator(const NeighborQuery* neighbor_query, const vec3<float>* query_points, unsigned int num_query_points,
                          QueryArgs qargs)
        : m_neighbor_query(neighbor_query), m_query_points(query_points), m_num_query_points(num_query_points),
          m_qargs(qargs), m_finished(false), m_cur_p(0)
    {
        m_iter = m_neighbor_query->queryWithArgs(m_query_points[m_cur_p], m_cur_p, m_qargs);
    }

    //! Empty Destructor
    ~NeighborQueryIterator() {}

    //! Indicate when done.
    bool end()
    {
        return m_finished;
    }

    //! Get an iterator for a specific query point by index.
    std::shared_ptr<NeighborQueryPerPointIterator> query(unsigned int i)
    {
        return m_neighbor_query->queryWithArgs(m_query_points[i], i, m_qargs);
    }

    //! Get the next element.
    NeighborBond next()
    {
        if (m_finished)
            return ITERATOR_TERMINATOR;
        NeighborBond nb;
        while (true)
        {
            while (!m_iter->end())
            {
                nb = m_iter->next();

                if (nb != ITERATOR_TERMINATOR)
                {
                    return nb;
                }
            }
            m_cur_p++;
            if (m_cur_p >= m_num_query_points)
                break;
            m_iter = m_neighbor_query->queryWithArgs(m_query_points[m_cur_p], m_cur_p, m_qargs);
        }
        m_finished = true;
        return ITERATOR_TERMINATOR;
    }

    ////! Generate a NeighborList from query.
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
    NeighborList* toNeighborList()
    {
        typedef tbb::enumerable_thread_specific<std::vector<NeighborBond>> BondVector;
        BondVector bonds;
        tbb::parallel_for(tbb::blocked_range<size_t>(0, m_num_query_points), [&](const tbb::blocked_range<size_t>& r) {
            BondVector::reference local_bonds(bonds.local());
            NeighborBond nb;
            for (size_t i(r.begin()); i != r.end(); ++i)
            {
                std::shared_ptr<NeighborQueryPerPointIterator> it = this->query(i);
                while (!it->end())
                {
                    nb = it->next();
                    // If we're excluding ii bonds, we have to check before adding.
                    if (nb != ITERATOR_TERMINATOR)
                    {
                        local_bonds.emplace_back(nb.id, nb.ref_id, nb.distance);
                    }
                }
            }
        });

        tbb::flattened2d<BondVector> flat_bonds = tbb::flatten2d(bonds);
        std::vector<NeighborBond> linear_bonds(flat_bonds.begin(), flat_bonds.end());
        tbb::parallel_sort(linear_bonds.begin(), linear_bonds.end(), compareNeighborBond);

        unsigned int num_bonds = linear_bonds.size();

        NeighborList* nl = new NeighborList();
        nl->resize(num_bonds);
        nl->setNumBonds(num_bonds, m_num_query_points, m_neighbor_query->getNPoints());
        size_t* neighbor_array(nl->getNeighbors());
        float* neighbor_weights(nl->getWeights());
        float* neighbor_distance(nl->getDistances());

        parallel_for(tbb::blocked_range<size_t>(0, num_bonds), [&](const tbb::blocked_range<size_t>& r) {
            for (size_t bond(r.begin()); bond < r.end(); ++bond)
            {
                neighbor_array[2 * bond] = linear_bonds[bond].id;
                neighbor_array[2 * bond + 1] = linear_bonds[bond].ref_id;
                neighbor_distance[bond] = linear_bonds[bond].distance;
            }
        });
        memset((void*) neighbor_weights, 1, sizeof(float) * linear_bonds.size());

        return nl;
    }

    static const NeighborBond ITERATOR_TERMINATOR; //!< The object returned when iteration is complete.

protected:
    const NeighborQuery* m_neighbor_query; //!< Link to the NeighborQuery object.
    const vec3<float> *m_query_points;           //!< Coordinates of the query point.
    unsigned int m_num_query_points; //!< The index of the query point.
    unsigned int cur_p;                    //!< The current index into the points (bounded by m_n_query_points).
    const QueryArgs m_qargs;  //!< The current query arguments
    std::shared_ptr<NeighborQueryPerPointIterator> m_iter;  //!< The per-point iterator being used.

    bool m_finished;    //!< Flag to indicate that iteration is complete (must be set by next on termination).
    bool m_exclude_ii; //!< Flag to indicate whether or not to include self bonds.
    unsigned int m_cur_p;  //!< The current particle under consideration.
};


// Dummy class to just contain minimal information and not actually query.
class RawPoints : public NeighborQuery
{
public:
    RawPoints();

    RawPoints(const box::Box& box, const vec3<float>* points, unsigned int n_points)
        : NeighborQuery(box, points, n_points)
    {}

    ~RawPoints() {}

    // dummy implementation for pure virtual function in the parent class
    virtual std::shared_ptr<NeighborQueryPerPointIterator> queryWithArgs(const vec3<float> query_point, unsigned int query_point_idx,
                                                         QueryArgs qargs) const
    {
        throw std::runtime_error("The queryArgs method is not implemented for RawPoints.");
    }
};

}; }; // end namespace freud::locality

#endif // NEIGHBOR_QUERY_H
