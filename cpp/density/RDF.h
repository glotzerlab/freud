// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef RDF_H
#define RDF_H

#include <memory>

#include "Box.h"
#include "NdHistogram.h"
#include "NeighborList.h"
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
    RDF(float rmax, float dr, float rmin = 0);

    //! Destructor
    ~RDF() {};

    //! Reset the RDF array to all zeros
    void reset();

    //! Compute the RDF
    void accumulate(box::Box& box, const freud::locality::NeighborList* nlist, const vec3<float>* ref_points,
                    unsigned int n_ref, const vec3<float>* points, unsigned int n_p);

    //! \internal
    //! helper function to reduce the thread specific arrays into one array
    void reduceRDF();

    //! Implementing pure virtual function from parent class.
    virtual void reduce()
    {
        reduceRDF();
    }

    //! Get a reference to the PCF array
    std::shared_ptr<float> getRDF()
    {
        return getPCF();
    }

    //! Get a reference to the r array
    std::shared_ptr<float> getR();

    //! Get a reference to the N_r array
    std::shared_ptr<float> getNr()
    {
        return reduceAndReturn(m_N_r_array);
    }

    unsigned int getNBins();

private:
    float m_rmax;         //!< Maximum r at which to compute g(r)
    float m_rmin;         //!< Minimum r at which to compute g(r)
    float m_dr;           //!< Step size for r in the computation
    unsigned int m_nbins; //!< Number of r bins to compute g(r) over

    std::shared_ptr<float> m_avg_counts;  //!< Bin counts that go into computing the RDF array
    std::shared_ptr<float> m_N_r_array;   //!< Cumulative bin sum N(r)
    std::shared_ptr<float> m_r_array;     //!< Array of r values that the RDF is computed at
    std::shared_ptr<float> m_vol_array;   //!< Array of volumes for each slice of r
    std::shared_ptr<float> m_vol_array2D; //!< Array of volumes for each slice of r
    std::shared_ptr<float> m_vol_array3D; //!< Array of volumes for each slice of r
};

}; }; // end namespace freud::density

#endif // RDF_H
