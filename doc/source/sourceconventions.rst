=======================
Source Code Conventions
=======================

The guidelines below should be followed for any new code added to freud.

Code Guidelines
===============

Code in freud should follow
`PEP8 <https://www.python.org/dev/peps/pep-0008/>`_, as well as the
following guidelines. Anything listed here takes precedence over PEP8,
but try to deviate as little as possible from PEP8. When in doubt,
follow these guidelines over PEP8.
If you are unsure if your code is pep8 compliant, you can use autopep8
and flake8 (or similar) to automatically update and check your code.

Naming Conventions
==================

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

Semicolons in Python
====================

Semicolons should not be used to mark the end of lines in Python.

Indentation
===========

-  Spaces, not tabs, must be used for indentation
-  *4 spaces* are required per level of indentation
-  *4 spaces* are *required*, not optional, for continuation lines
-  There should be no whitespace at the end of lines in the file.
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

Formatting Long Lines
=====================

All code lines should be hand-wrapped so that they are no more than
*79 characters* long. Simply break any excessively long line of code at any
natural breaking point to continue on the next line.

.. code-block:: c++

    cout << "This is a really long message, with "
         << message.length()
         << "Characters in it:"
         << message << endl;

Try to maintain some element of beautiful symmetry in the way the line is
broken. For example, the *above* long message is preferred over the below:

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

Documentation Comments
======================

-  Documentation should be included at the Python-level in the Cython
   wrapper.
-  Every class, member variable, function, function parameter, macro,
   etc. must be documented with *Python docstring* comments which will
   be converted to documentation with sphinx.
-  See the `sphinx documentation <http://www.sphinx-doc.org/en/stable/index.html>`_
   for more information
-  If you copy an existing file as a template, do not leave the
   existing documentation comments there. They apply to the original
   file, not your new one!
-  The best advice that can be given is to write the documentation
   comments *first* and the actual code *second*. This allows one to
   formulate their thoughts and write out in English what the code is
   going to be doing. After thinking through that, writing the actual
   code is often *much easier*, plus the documentation left for future
   developers to read is top-notch.
-  Good documentation comments are best demonstrated with an in-code
   example.
