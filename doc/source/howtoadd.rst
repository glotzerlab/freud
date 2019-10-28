===================
How to Add New Code
===================

This document details the process of adding new code into **freud**.

Does my code belong in **freud**?
=================================

The **freud** library is not meant to simply wrap or augment external Python
libraries. A good rule of thumb is *if the code I plan to write does not
require C++, it does not belong in freud*. There are, of course, exceptions.

Create a new branch
===================

You should branch your code from :code:`master` into a new branch. Do not add
new code directly into the :code:`master` branch.

Add a New Module
================

If the code you are adding is in a *new* module, not an existing module, you must do the following:

- Create :code:`cpp/moduleName` folder

- Edit :code:`freud/__init__.py`

  - Add :code:`from . import moduleName` so that your module is imported by default.

- Edit :code:`freud/_freud.pyx`

  - Add :code:`include "moduleName.pxi"`. This must be done to have freud include your Python-level code.

- Create :code:`freud/moduleName.pxi` file

  - This will house the python-level code.
  - If you have a .pxd file exposing C++ classes, make sure to import that:

::

   cimport freud._moduleName as moduleName

- Create :code:`freud/moduleName.py` file

  - Make sure there is an import for each C++ class in your module:

::

    from ._freud import MyC++Class

- Create :code:`freud/_moduleName.pxd`

  - This file will expose the C++ classes in your module to python.

- Edit :code:`setup.py`

  - Add :code:`cpp/moduleName` to the :code:`includes` list.
  - If there are any helper cc files that will not have a corresponding Cython class, add those files to the :code:`sources` list inside the :code:`extensions` list.

- Add line to :code:`doc/source/index.rst`

  - Make sure your new module is referenced in the documentation.

- Create :code:`doc/source/moduleName.rst`

Add to an Existing Module
=========================

To add a new class to an existing module, do the following:

- Create :code:`cpp/moduleName/SubModule.h` and
  :code:`cpp/moduleName/SubModule.cc`

  - New classes should be grouped into paired :code:`.h`, :code:`.cc` files.
    There may be a few instances where new classes could be added to an
    existing :code:`.h`, :code:`.cc` pairing.

- Edit :code:`freud/moduleName.py` file

  - Add a line for each C++ class in your module:

::

    from ._freud import MyC++Class

- Expose C++ class in :code:`freud/_moduleName.pxd`

- Create Python interface in :code:`freud/moduleName.pxi`

You must include sphinx-style documentation and unit tests.

- Add extra documentation to :code:`doc/source/moduleName.rst`

- Add unit tests to :code:`freud/tests`
