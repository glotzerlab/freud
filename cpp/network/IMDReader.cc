#include <boost/python.hpp>
#include <boost/shared_array.hpp>
#include "IMDReader.h"
#include "vmdsock.h"
#include "imd.h"


    IMDReader::IMDReader(unsigned int N)
{
    unsigned int m_N = N;
}

bool IMDReader::connect(const std::string& hostname, unsigned int port)
{
    // connect to the given host. Issues vmdsock_init, _create, and _connect.  Then imd_handshake and  send the go message. Returns true if the connection is made
    int err = 0;
    // m_error holds the error information of the run. By default, no connection means no error
    bool m_error = false;

    // initialize the receiving socket
    vmdsock_init();
    m_sock = vmdsock_create();

    // here check for errors ...
    
    // connect the socket and start listening for connections on that port
    vmdsock_connect(m_sock, hostname.c_str(), port);

    // create the connection
    if (imd_handshake(m_sock))
        {
        vmdsock_destroy(m_sock);
        m_sock = NULL;
        m_error = true;
        return;
        }
    else
        {
            m_error = false;
        }

}

void IMDReader::disconnect()
{
    // disconnect. Send the disconnect message and then vmdsock_destroy.
}

void IMDReader::set_trate(trate)
{
    // set the transmission rate. calls imd_trate
}

void IMDReader::pause()
{
    // send the pause command
    if(imd_pause(m_sock))
    {
        m_error = true;
    }
}

void IMDReader::go()
{
    if (imd_go(m_sock))
        m_error = true;
}

bool IMDReader::good()
{
    return !m_error
}

void IMDReader::receive_fcoords()
{
    imd_recv_fcoord(m_sock, m_N, float *coords);
}

bool IMDReader::poll()
{
    if(good())
    {
        int length;
        int res = vmdsock_selread(m_sock, timeout);
        if (res == -1)
        {
            m_error = true;
            return
        }
        if (res == 1)
        {
            IMDType header = imd_recv_header(m_sock, &length);
            
            switch(header)
            {
                case IMD_FCOORDS:
                    receive_fcoords();
                    break;
                case IMD_GO:
                    go();
                    break;
                case IMD_PAUSE:
                    pause();
                    break;
                case IMD_TRATE:
                    set_trate();
                    break;
                case IMD_DISCONNECT:
                    disconnect();
                    break;
                default:
                    m_error = true;
                    break;

            }

        }

     }

}

void export_IMDReader()
    {   
    class_<IMDReader>("IMDReader", init<int>())
        ;   
    }   
