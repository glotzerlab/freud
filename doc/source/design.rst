======
Design
======

Vision
------

Freud is designed to be:

1. Powerful
2. Flexible
3. Maintainable

Powerful
~~~~~~~~

The amount of data produced by simulations is always increasing. By
being powerful, Freud allows users to analyze their simulation data as
fast as possible so that it can be used in real-time visualization and
on-line simulation analysis.

Flexible
~~~~~~~~

The number of simulation packages, analysis packages, and other software
packages keeps growing. Rather than attempt to understand and interact
with all of these packages, Freud achieves flexibility by providing a
simple Python interface and making no assumptions regarding data,
operating on and returning numpy arrays to the user.

Maintainable
~~~~~~~~~~~~

Code which cannot be maintained is destined for obscurity. In order to
be maintainable, Freud uses git for version control; bit bucket for code
hosting, issue tracking; the PEP8 standard for code, stressing
explicitly written code which is easy to read.

Language choices
----------------

Freud is written in two languages: Python and C++. C++ allows for
powerful, fast code execution while Python allows for easy, flexible
use. Intel Thread Building Blocks parallelism provides further power to
C++ code. The C++ code is wrapped with Cython, allowing for user
interaction in Python. NumPy provides the basic data structures in
Freud, which are commonly used in other Python plotting libraries and
packages.

Code Guidlines
--------------

Code in Freud should follow
`PEP8 <https://www.python.org/dev/peps/pep-0008/>`__, as well as the
following guidelines. Anything listed here takes precedence over PEP8,
but we try to deviate as little as possible from PEP8. *When in doubt,
follow the guidelines!*

Python and Cython naming conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Variables should be named using ``lower_case_with_underscores``
-  Functions and methods should be named using
   ``lowerCaseWithNoUnderscores``
-  Classes should be named using ``CapWords``

Example:

::

    class FreudClass(object):
        def __init__(self):
            pass
        def calcSomething(self, position_i, orientation_i, position_j, orientation_j):
            r_ij = position_j - position_i
            theta_ij = calcOrientationThing(orientation_i, orientation_j)
        def calcOrientationThing(self, orientation_i, orientation_j):
            ...

C++ naming conventions
~~~~~~~~~~~~~~~~~~~~~~

To intuitively distinguish between C++ and Python code, the following
conventions should be used:

-  Variables are named with ``lower_case_with_underscores``
-  Functions and methods are named with ``lowerCaseWithNoUnderscores``
-  Classes are named with ``CapWords``

Example:

::

    class FreudCPPClass
        {
        FreudCPPClass()
            {
            }
        computeSomeValue(int variable_a, float variable_b)
            {
            // do some things in here
            }
        };

Make things explicit, not automatic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While it is tempting to make your code do things "automatically", such
as have a calculate method find all ``_calc`` methods in a class, call
them, and add their returns to a dictionary to return to the user, it is
preferred in Freud to do this explicitly. This helps avoid issues in
debugging and undocumented behavior:

::

    #!python
    # this is bad
    class SomeFreudClass(object):
        def __init__(self, **kwargs):
            for key in kwargs.keys:
                setattr(self, key, kwargs[key])

    # this is good
    class SomeOtherFreudClass(object):
        def __init__(self, x=None, y=None):
            self.x = x
            self.y = y

Code duplication
~~~~~~~~~~~~~~~~

When possible, code should not be duplicated. However, being explicit is
more important. In Freud this translates to many of the inner loops of
functions being very similar:

::

    #!c++
    // somewhere deep in a function_a
            for (int i = 0; i < n; i++)
                {
                vec3[float] pos_i = position[i];
                    for (int j = 0; j < n; j++)
                        {
                        pos_j = = position[j];
                        // more calls here
                        }
                }

    #!c++
    // somewhere deep in a function_b
            for (int i = 0; i < n; i++)
                {
                vec3[float] pos_i = position[i];
                    for (int j = 0; j < n; j++)
                        {
                        pos_j = = position[j];
                        // more calls here
                        }
                }

While it *might* be possible to figure out a way to create a base C++
class all such classes inherit from, run through positions, call a
calculation, and return, this would be rather complicated. Additionally,
any changes to the internals of the code, and may result in performance
penalties, difficulty in debugging, etc. As before, being explicit is
better.

However, if you have a class which has a number of methods, each of
which requires the calling of a function, this function should be
written as its own method, instead of being copy-pasted into each
method, as is typical in object-oriented programming.

Python vs. Cython vs. C++
~~~~~~~~~~~~~~~~~~~~~~~~~

Freud is meant to leverage the power of C++ code imbued with parallel
processing power from TBB with the ease of writing Python code. The bulk
of your calculations should take place in C++, as shown in the snippet
below

::

    #!python
    # this is bad
    def heavyLiftingInFreud(positions):
        # check that positions are fine
        for i, pos_i in enumerate(positions):
            for j, pos_j in enumerate(positions):
                if i != j:
                    r_ij = pos_j - pos_i
                    ...
                    computed_array[i] += some_val
        return computed_array

    # this is good
    def callC++ForHeavyLifting(positions):
        # check that positions are fine
        c++_heavy_function(computed_array, positions, len(pos))
        return computed_array

    #!c++

    void c++HeavyLifting(float* computed_array,
                         float* positions,
                         int n)
        {
        for (int i = 0; i < n; i++)
            {
            for (int j = 0; j < n; j++)
                {
                if (i != j)
                    {
                    r_ij = pos_j - pos_i;
                    ...
                    computed_array[i] += some_val;
                    }
                }
            }
        }

However, some functions may be necessary to write at the Python level
due to a Python library not having an equivalent C++ library, complexity
of coding, etc. In this case, the code should be written in Cython and a
*reasonable* attempt to optimize the code should be made.

Semicolons in Python
~~~~~~~~~~~~~~~~~~~~

Semicolons should not be used to mark the end of lines in Python

Indentation
~~~~~~~~~~~

-  Spaces, not tabs, must be used for indentation
-  *4 spaces* are required per level of indentation
-  *4 spaces* are *required*, not optional, for continuation lines
-  There should be no whitespace at the end of lines in the file.
-  C++ code should follow `Whitesmith's
   style <http://en.wikipedia.org/wiki/Indent_style#Whitesmiths_style>`__.
   An extended set of examples follows:

::

    #!c++
    class SomeClass
        {
        public:
            SomeClass();
            int SomeMethod(int a);
        private:
            int m_some_member;
        };

    // indent function bodies
    int SomeClass::SomeMethod(int a)
        {
        // indent loop bodies
        while (condition)
            {
            b = a + 1;
            c = b - 2;
            }

        // indent switch bodies and the statements inside each case
        switch (b)
            {
            case 0:
                c = 1;
                break;
            case 1:
                c = 2;
                break;
            default:
                c = 3;
                break;
            }

        // indent the bodies of if statements
        if (something)
            {
            c = 5;
            b = 10;
            }
         else if (something_else)
            {
            c = 10;
            b = 5;
            }
         else
            {
            c = 20;
            b = 6;
            }

        // omitting the braces is fine if there is only one statement in a body (for loops, if, etc.)
        for (int i = 0; i < 10; i++)
            c = c + 1;

        return c;
        // the nice thing about this style is that every brace lines up perfectly with it's mate
        }

-  Documentation comments and items broken over multiple lines should be
   *aligned* with spaces

::

    #!c++
    class SomeClass
        {
        private:
            int m_some_member;        //!< Documentation for some_member
            int m_some_other_member;  //!< Documentation for some_other_member
        };

    template<class BlahBlah> void some_long_func(BlahBlah with_a_really_long_argument_list,
                                                 int b,
                                                 int c);

-  TBB sections should use lambdas, not templates

::

    #!c++
    void someC++Function(float some_var,
                         float other_var)
        {
        // code before parallel section
        parallel_for(blocked_range<size_t>(0,n),
            [=] (const blocked_range<size_t>& r)
                {
                // do stuff
                });

Formatting Long Lines
~~~~~~~~~~~~~~~~~~~~~

-  All code lines should be hand-wrapped so that they are no more than
   *79 characters* long
-  Simply break any excessively long line of code at any natural
   breaking point to continue on the next line

::

    #!c++
    cout << "This is a really long message, with "
         << message.length()
         << "Characters in it:"
         << message << endl;

-  Try to maintain some element of beautiful symmetry in the way the
   line is broken. For example, the *above* long message is preferred
   over the below:

::

    #!c++
    cout << "This is a really long message, with " << message.length() << "Characters in it:"
       << message << endl;

-  There are *special rules* for function definitions and/or calls
-  If the function definition (or call) cleanly fits within the 120
   character limit, leave it all on one line

::

    #!c++
    int some_function(int arg1, int arg2)

-  (option 1) If the function definition (or call) goes over the limit,
   you may be able to fix it by simply putting the template definition
   on the previous line:

::

    #!c++
    // go from
    template<class Foo, class Bar> int some_really_long_function_name(int with_really_long, Foo argument, Bar lists)
    // to
    template<class Foo, class Bar>
    int some_really_long_function_name(int with_really_long, Foo argument, Bar lists)

-  (option 2) If the function doesn't have a template specifier, or
   splitting at that point isn't enough, split out each argument onto a
   separate line and align them.

::

    #!c++
    // go from
    int someReallyLongFunctionName(int with_really_long_arguments, int or, int maybe, float there, char are, int just, float a, int lot, char of, int them)
    // to
    int someReallyLongFunctionName(int with_really_long_arguments,
                                   int or,
                                   int maybe,
                                   float there,
                                   char are,
                                   int just,
                                   float a,
                                   int lot,
                                   char of,
                                   int them)

Documentation Comments
~~~~~~~~~~~~~~~~~~~~~~

-  Documentation should be included at the Python-level in the Cython
   wrapper.
-  Every class, member variable, function, function parameter, macro,
   etc. *MUST* be documented with *python docstring* comments which will
   be converted to documentation with *sphinx*.
-  See http://www.sphinx-doc.org/en/stable/index.html
-  If you copy an existing file as a template, *DO NOT* simply leave the
   existing documentation comments there. They apply to the original
   file, not your new one!
-  The best advice that can be given is to write the documentation
   comments *FIRST* and the *actual code* *second*. This allows one to
   formulate their thoughts and write out in English what the code is
   going to be doing. After thinking through that, writing the actual
   code is often *much easier*, plus the documentation left for future
   developers to read is top-notch.
-  Good documentation comments are best demonstrated with an in-code
   example.
