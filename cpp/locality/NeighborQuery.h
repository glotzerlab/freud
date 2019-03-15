// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef NEIGHBOR_QUERY_H
#define NEIGHBOR_QUERY_H

#include "Box.h"
#include "NeighborList.h"
#include <stdexcept>
#include <memory>
#include <tbb/tbb.h>

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
struct NeighborPoint {
    NeighborPoint() : id(0), ref_id(0), distance(0) { }

    NeighborPoint(unsigned int id, unsigned int ref_id, float d) : id(id), ref_id(ref_id), distance(d) { }

    //! Equality checks both id and distance.
    bool operator== (const NeighborPoint &n)
        {
        return (id == n.id) && (ref_id == n.ref_id) && (distance == n.distance);
        }

    //! Default comparator of points is by distance.
    /*! This form of comparison allows easy sorting of nearest neighbors by
     *  distance
     */
    bool operator< (const NeighborPoint &n) const
        {
        return distance < n.distance;
        }

    unsigned int id;
    unsigned int ref_id;
    float distance;
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
        NeighborQuery()
            {
            }

        //! Constructor
        NeighborQuery(const box::Box &box, const vec3<float> *ref_points, unsigned int Nref) :
            m_box(box), m_ref_points(ref_points), m_Nref(Nref)
            {
            }

        //! Empty Destructor
        virtual ~NeighborQuery() {}

        //! Given a point, find the k elements of this data structure
        //  that are the nearest neighbors for each point.
        virtual std::shared_ptr<NeighborQueryIterator> query(const vec3<float> *points, unsigned int N, unsigned int k) const = 0;

        //! Given a point, find all elements of this data structure
        //  that are within a certain distance r.
        virtual std::shared_ptr<NeighborQueryIterator> queryBall(const vec3<float> *points, unsigned int N, float r) const = 0;

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
        const vec3<float> operator[] (unsigned int index) const
            {
            if (index >= m_Nref)
                {
                throw std::runtime_error("NeighborQuery attempted to access a point with index >= Nref.");
                }
            return m_ref_points[index];
            }

    protected:
        const box::Box m_box;             //!< Simulation box where the particles belong
        const vec3<float> *m_ref_points;  //!< Reference point coordinates
        unsigned int m_Nref;              //!< Number of reference points
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
class NeighborQueryIterator {
    public:
        //! Nullary constructor for Cython
        NeighborQueryIterator()
            {
            }

        //! Constructor
        NeighborQueryIterator(const NeighborQuery* neighbor_query,
                const vec3<float> *points, unsigned int N) :
            m_neighbor_query(neighbor_query), m_points(points), m_N(N), cur_p(0), m_finished(false)
            {
            }

        //! Empty Destructor
        virtual ~NeighborQueryIterator() {}

        //! Indicate when done.
        virtual bool end() { return m_finished; }

        //! Replicate this class's query on a per-particle basis.
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
         */
        NeighborList *toNeighborList()
            {
            typedef tbb::enumerable_thread_specific<std::vector<std::pair<size_t, size_t>>> BondVector;
            BondVector bonds;
            tbb::parallel_for(tbb::blocked_range<size_t>(0, m_N),
                [&] (const tbb::blocked_range<size_t> &r)
                {
                BondVector::reference local_bonds(bonds.local());
                NeighborPoint np;
                for (size_t i(r.begin()); i != r.end(); ++i)
                    {
                    std::shared_ptr<NeighborQueryIterator> it = query(i);
                    while (!it->end())
                        {
                        np = it->next();
                        // Swap ref_id and id order for backwards compatibility.
                        // I NEED TO MAKE THE QUERY METHOD RETURN THINGS MORE APPROPRIATELY, right now I'm forced to manually replace the id with i.
                        local_bonds.emplace_back(np.ref_id, i);
                        }
                    // Remove the last item, which is just the terminal sentinel value.
                    local_bonds.pop_back();
                    }
                });
                
            tbb::flattened2d<BondVector> flat_bonds = tbb::flatten2d(bonds);
            std::vector<std::pair<size_t, size_t>> linear_bonds(flat_bonds.begin(), flat_bonds.end());
            tbb::parallel_sort(linear_bonds.begin(), linear_bonds.end());

            unsigned int num_bonds = linear_bonds.size();

            NeighborList *nl = new NeighborList();
            nl->resize(num_bonds);
            nl->setNumBonds(num_bonds, m_neighbor_query->getNRef(), m_N);
            size_t *neighbor_array(nl->getNeighbors());
            float *neighbor_weights(nl->getWeights());

            parallel_for(tbb::blocked_range<size_t>(0, num_bonds),
                [&] (const tbb::blocked_range<size_t> &r)
                {
                for (size_t bond(r.begin()); bond < r.end(); ++bond)
                    {
                    neighbor_array[2*bond] = linear_bonds[bond].first;
                    neighbor_array[2*bond+1] = linear_bonds[bond].second;
                    }
                });
            memset((void*) neighbor_weights, 1, sizeof(float)*linear_bonds.size());

            return nl;
            }

        static const NeighborPoint ITERATOR_TERMINATOR; //!< The object returned when iteration is complete.

    protected:
        const NeighborQuery *m_neighbor_query; //!< Link to the NeighborQuery object.
        const vec3<float> *m_points;         //!< Coordinates of query points.
        unsigned int m_N;                    //!< Number of points.
        unsigned int cur_p;                  //!< The current index into the points (bounded by m_N).

        unsigned int m_finished;             //!< Flag to indicate that iteration is complete (must be set by next on termination).
};

}; }; // end namespace freud::locality

#endif // NEIGHBOR_QUERY_H
