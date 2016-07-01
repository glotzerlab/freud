#ifndef __DCDLOADER__H__
#define __DCDLOADER__H__

#include <memory>

#include "molfile/molfile_plugins.h"
#include "trajectory.h"

#include <cassert>
#include <string>

#include "ScopedGILRelease.h"

/*! \file DCDLoader.h
    \brief DCD file reader
*/

namespace freud { namespace trajectory {

//! Class for loading DCD files into freud
/*! The structure information is assumed to have been read in elsewhere.

    Every call to readNextStep() will fill out the existing Nx3 numpy array. Users in python will have to be aware
    of this and plan accordingly (specifically TrajectoryXMLDCD will make a copy so that users aren't confused)

    The values read in can be accessed with getPoints() and getBox()

    jumpToFrame() is smart in that it can be called as many times as needed and it will not re-read the file
    if it only has to advance. If a previous frame is selected, however, jumpToFrame has no choice but to close
    and reopen the file and reread it from the beginning.
    */
class DCDLoader
    {
    public:
        //! Constructs the loader and associates it to the given file
        DCDLoader(const std::string &dcd_fname);
        //! Frees all dynamic memory
        ~DCDLoader();

        //! Jumps to a particular frame number in the file
        void jumpToFrame(int frame);

        //! Read the next step in the file
        void readNextFrame();

        //! Jump to a particular frame number in the file (call only from python)
        // void jumpToFramePy(int frame)
        //     {
        //     util::ScopedGILRelease gil;
        //     jumpToFrame(frame);
        //     }

        // //! Read the next step in the file (call only from python)
        // void readNextFramePy()
        //     {
        //     util::ScopedGILRelease gil;
        //     readNextFrame();
        //     }

        // //! Access the points read by the last step
        std::shared_ptr<float> getPoints() const
            {
            // allocate the memory for the points
            // std::vector<intp> dims(2);
            // dims[0] = getNumParticles();
            // dims[1] = 3;

            // float *arr = m_points.get();
            // return num_util::makeNum(arr, dims);
            return m_points;
            }

        //! Get the box
        const Box& getBox() const
            {
            return m_box;
            }

        //! Get the number of particles
        unsigned int getNumParticles() const
            {
            assert(m_dcd);
            return m_dcd->natoms;
            }

        //! Get the last frame read
        unsigned int getLastFrameNum() const
            {
            assert(m_dcd);
            // the frame counter is advanced when read, so subtract 1 to get the last one read
            return m_dcd->setsread-1;
            }

        //! Get the number of frames
        unsigned int getFrameCount() const
            {
            assert(m_dcd);
            return m_dcd->nsets;
            }

        //! Get the original filename
        std::string getFileName() const
            {
            return m_fname;
            }

        //! Get the time step
        unsigned int getTimeStep() const
            {
            return m_time_step;
            }


    private:
        std::string m_fname;                        //!< File name of the DCD file
        Box m_box;                                  //!< The box read from the last readNextStep()
        std::shared_ptr<float> m_points;        //!< Points read during the last readNextStep()
        unsigned int m_time_step;                   //!< Time step value read

        //! Keep track of the dcd file
        dcdhandle *m_dcd;

        //! Keep track of the dcd plugin
        molfile_plugin_t *dcdplugin;

        //! Helper function to start loading the dcd file
        void loadDCD();
    };

}; }; // end namespace freud::trajectory

#endif
