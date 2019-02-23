// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef PMFT_H
#define PMFT_H

#include <memory>
#include <ostream>
#include <tbb/tbb.h>

#include "Box.h"
#include "VectorMath.h"
#include "LinkCell.h"

#include "Index1D.h"

/*! \internal
    \file PMFT.h
    \brief Declares base class for all PMFT classes
*/

namespace freud { namespace pmft {

//! Computes the PMFT for a given set of points
/*! The PMFT class is an abstract class providing the basis for all classes calculating PMFTs for specific
 *  dimensional cases. The PMFT class defines some of the key interfaces required for all PMFT classes, such
 *  as the ability to access the underlying PCF and box. Many of the specific methods must be implemented by
 *  subclasses that account for the proper set of dimensions.The required functions are implemented as pure
 *  virtual functions here to enforce this.
*/
class PMFT
    {
    public:
        //! Constructor
        PMFT();

        //! Destructor
        virtual ~PMFT();

        //! Get the simulation box
        const box::Box& getBox() const
            {
            return m_box;
            }

        //! Reset the PCF array to all zeros
        virtual void reset() = 0;

        //! \internal
        //! helper function to reduce the thread specific arrays into one array
        //! Must be implemented by subclasses
        virtual void reducePCF() = 0;

        //! Get a reference to the PCF array
        std::shared_ptr<float> getPCF();

        //! Get a reference to the bin counts array
        std::shared_ptr<unsigned int> getBinCounts();

        float getRCut()
            {
            return m_r_cut;
            }

    protected:
        box::Box m_box;                    //!< Simulation box where the particles belong
        float m_r_cut;                     //!< r_cut used in cell list construction
        unsigned int m_frame_counter;      //!< Number of frames calculated
        unsigned int m_n_ref;              //!< The number of reference points
        unsigned int m_n_p;                //!< The number of points
        bool m_reduce;                     //!< Whether or not the PCF needs to be reduced

        std::shared_ptr<float> m_pcf_array;            //!< Array of PCF computed
        std::shared_ptr<unsigned int> m_bin_counts;    //!< Counts for each bin
        tbb::enumerable_thread_specific<unsigned int *> m_local_bin_counts; //!< Thread local bin counts for TBB parallelism

    private:
    };

}; }; // end namespace freud::pmft

#endif // PMFT_H
