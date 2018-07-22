=======================
Source Code Conventions
=======================

The guidelines below should be followed for any new code added to freud.
This guide is separated into three sections, one for guidelines common
to Python and C++, one for Python alone, and one for C++.

Both
====

Naming Conventions
------------------

The following conventions should apply to Python, Cython, and C++ code.

-  Variable names use :code:`lower_case_with_underscores`
-  Function and method names use :code:`lowerCaseWithNoUnderscores`
-  Class names use :code:`CapWords`

Python example:

.. code-block:: python

    class FreudClass(object):
        def __init__(self):
            pass
        def calcSomething(self, position_i, orientation_i, position_j, orientation_j):
            r_ij = position_j - position_i
            theta_ij = calcOrientationThing(orientation_i, orientation_j)
        def calcOrientationThing(self, orientation_i, orientation_j):
            ...

C++ example:

.. code-block:: c++

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

Indentation
-----------

-  Spaces, not tabs, must be used for indentation
-  *4 spaces* are required per level of indentation and continuation lines
-  There should be no whitespace at the end of lines in the file.
-  Documentation comments and items broken over multiple lines should be
   *aligned* with spaces

.. code-block:: c++

    class SomeClass
        {
        private:
            int m_some_member;        //!< Documentation for some_member
            int m_some_other_member;  //!< Documentation for some_other_member
        };

    template<class BlahBlah> void some_long_func(BlahBlah with_a_really_long_argument_list,
                                                 int b,
                                                 int c);

Formatting Long Lines
---------------------

All code lines should be hand-wrapped so that they are no more than
*79 characters* long. Simply break any excessively long line of code at any
natural breaking point to continue on the next line.

.. code-block:: c++

    cout << "This is a really long message, with "
         << message.length()
         << "Characters in it:"
         << message << endl;

Try to maintain some element of symmetry in the way the line is broken.
For example, the *above* long message is preferred over the below:

.. code-block:: c++

    cout << "This is a really long message, with " << message.length() << "Characters in it:"
       << message << endl;

There are *special rules* for function definitions and/or calls:

-  If the function definition (or call) cleanly fits within the
   character limit, leave it all on one line

.. code-block:: c++

    int some_function(int arg1, int arg2)

-  (Option 1) If the function definition (or call) goes over the limit,
   you may be able to fix it by simply putting the template definition
   on the previous line:

.. code-block:: c++

    // go from
    template<class Foo, class Bar> int some_really_long_function_name(int with_really_long, Foo argument, Bar lists)
    // to
    template<class Foo, class Bar>
    int some_really_long_function_name(int with_really_long, Foo argument, Bar lists)

-  (Option 2) If the function doesn't have a template specifier, or
   splitting at that point isn't enough, split out each argument onto a
   separate line and align them.

.. code-block:: c++

    // Instead of this...
    int someReallyLongFunctionName(int with_really_long_arguments, int or, int maybe, float there, char are, int just, float a, int lot, char of, int them)

    // ...use this.
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

Python
======

Code in freud should follow
`PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_, as well as the
following guidelines. Anything listed here takes precedence over PEP 8,
but try to deviate as little as possible from PEP 8. When in doubt,
follow these guidelines over PEP 8.

If you are unsure if your code is PEP 8 compliant, you can use autopep8
and flake8 (or similar) to automatically update and check your code.


Source
------

- All code should be contained in Cython files
- Python .py files are reserved for module level docstrings and minor
  miscellaneous tasks for, *e.g*, backwards compatibility.
- Semicolons should not be used to mark the end of lines in Python.


Documentation Comments
----------------------

-  Documentation is generated using `sphinx <http://www.sphinx-doc.org/en/stable/index.html>`_.
-  The documentation should be written according to the `Google Python Style Guide <https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings>`_.
-  A few specific notes:

   - The shapes of NumPy arrays should be documented as part of the type in the
     following manner:
     :code:`points ((N, 4) (:py:class:np.ndarray)): The points...`.
   - Constructors should be documented at the class level.
   - Class attributes (*including properties*) should be documented as class
     attributes within the class-level docstring.
   - Optional arguments should be documented as such within the type after the
     actual type, and the default value should be included within the
     description *e.g.*,
     :code:`r_max (float, optional): ... If None (the default),
     number is inferred...`.
   - Properties that are settable should be documented the same way as optional
     arguments: :code:`Lx (float, settable): Length in x`.

-  All docstrings should be contained within the Cython files except module
   docstrings, which belong in the Python code.
-  If you copy an existing file as a template, **make sure to modify the comments
   to reflect the new file**.
-  Good documentation comments are best demonstrated with an in-code
   example. Liberal addition of examples is encouraged.

CPP
===

Indentation
-----------

-  C++ code should follow `Whitesmith's
   style <https://en.wikipedia.org/wiki/Indentation_style#Whitesmiths_style>`__.
   An extended set of examples follows:

.. code-block:: c++

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
        // the nice thing about this style is that every brace lines up perfectly with its mate
        }

-  TBB sections should use lambdas, not templates

.. code-block:: c++

    void someC++Function(float some_var,
                         float other_var)
        {
        // code before parallel section
        parallel_for(blocked_range<size_t>(0,n),
            [=] (const blocked_range<size_t>& r)
                {
                // do stuff
                });

Documentation Comments
----------------------

-  Documentation should be written in doxygen.
