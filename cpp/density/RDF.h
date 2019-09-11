// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef RDF_H
#define RDF_H

#include <memory>

#include "Box.h"
#include "NdHistogram.h"
#include "NeighborList.h"
#include "NeighborQuery.h"
#include "PMFT.h"
#include "ThreadStorage.h"
#include "VectorMath.h"

/*! \file RDF.h
    \brief Routines for computing radial density functions.
*/

namespace freud { namespace density {
class RDF : public util::NdHistogram
{
public:
    //! Constructor
    RDF(float r_max, float dr, float r_min = 0);

    //! Destructor
    ~RDF() {};

    //! Reset the RDF array to all zeros
    void reset();

    //! Compute the RDF
    void accumulate(const freud::locality::NeighborQuery* neighbor_query,
                    const vec3<float>* query_points, unsigned int n_query_points,
                    const freud::locality::NeighborList* nlist, freud::locality::QueryArgs qargs);

    //! \internal
    //! helper function to reduce the thread specific arrays into one array
    void reduceRDF();

    //! Implementing pure virtual function from parent class.
    virtual void reduce()
    {
        reduceRDF();
    }

    //! Get a reference to the PCF array
    const util::ManagedArray<float> &getRDF()
    {
        return getPCF();
    }

    //! Get a reference to the r array
    const util::ManagedArray<float> &getR();

    //! Get a reference to the N_r array.
    /*! Mathematically, m_N_r_array[i] is the average number of points
     * contained within a ball of radius m_r_array[i]+dr/2 centered at a given
     * query_point, averaged over all query_points.
     */
    const util::ManagedArray<float> &getNr()
    {
        return reduceAndReturn(m_N_r_array);
    }

    unsigned int getNBins();

private:
    float m_r_max;         //!< Maximum r at which to compute g(r)
    float m_r_min;         //!< Minimum r at which to compute g(r)
    float m_dr;           //!< Step size for r in the computation
    unsigned int m_nbins; //!< Number of r bins to compute g(r) over

    util::ManagedArray<float> m_avg_counts;  //!< Bin counts that go into computing the RDF array
    util::ManagedArray<float> m_N_r_array;   //!< Cumulative bin sum N(r)
    util::ManagedArray<float> m_r_array;     //!< Array of r values that the RDF is computed at
    util::ManagedArray<float> m_vol_array;   //!< Array of volumes for each slice of r
    util::ManagedArray<float> m_vol_array2D; //!< Array of volumes for each slice of r
    util::ManagedArray<float> m_vol_array3D; //!< Array of volumes for each slice of r
};

}; }; // end namespace freud::density

#endif // RDF_H
