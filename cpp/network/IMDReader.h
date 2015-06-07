#include <boost/python.hpp>
#include <boost/shared_array.hpp>
#include "HOOMDMath.h"
#include "VectorMath.h"

// declare only once
#ifndef __IMDREADER_H__
#define __IMDREADER_H__

class IMDReader
{
    public:
        IMDReader(unsigned int N);
        // destroy sockets. calls disconnect if we are not yet disconnected
        ~IMDReader()

        // connect to the given host. Issues vmdsock_init, _create, and _connect.  Then imd_handshake and  send the go message. Returns true if the connection is made.
        bool connect(hostname, port);

        // disconnect. Send the disconnect message and then vmdsock_destroy.
        void disconnect();

        // set the transmission rate. calls imd_trate
        void set_trate(trate);

        // send the pause command
        void pause();

        // send the go command
        void go();

        // return true if the connection is still valid. False if there is an error. Alternately, you could use exceptions in each of the methods. 
        // But a query method allows users to check for errors when it is convenient and not have to wrap everything in a try/except. TCP connections can drop at any time.
        bool good();

        // Poll the connection, returning false if there is no new data available (vmdsock_selread with timeout 0 - see IMDInterface.cc). 
        // Return true if a new system state was waiting and read. Poll stores the positions in a buffer.
        bool poll();

        void receive_fcoords();

        // Return the positions as a Nx3 numpy array
        boost::python::numeric::array get_position();
    
    private:
        boost::shared_array<vec3<float> > m_positions;
        unsigned int N;
};

//! Exports the IMDReader class to python
void export_IMDReader();

#endif
