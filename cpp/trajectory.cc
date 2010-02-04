#include <iostream>
#include <boost/python.hpp>

using namespace std;
using namespace boost::python;

void hello()
    {
    cout << "hello." << endl;
    }

BOOST_PYTHON_MODULE(_trajectory)
    {
    def("hello", &hello);
    }


