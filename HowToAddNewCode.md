# How to add new code to freud

This document details the process of adding new code into freud.

## Does my code belong in freud?

* Does this code simply wrap or augment an external python library?

If you answered "yes" to the question above, your code *probably* does not belong in freud. A good rule of thumb
is *if the code I plan to write does not require C++, it does not belong in freud*.

There are, of course, exceptions.

## Create a new branch

You should branch your code from `master` into a new branch. Do not add new code directly into the `master` branch.

## Add a new module

If the code you are adding is in a *new* module, not an existing module, you must do the following:

### Add lines to `cpp/CMakeLists.txt`

1. Add `${CMAKE_CURRENT_SOURCE_DIR}/moduleName` to `include_directories`.
2. Add `moduleName/SubModule.cc` and `moduleName/SubModule.h` to the `FREUD_SOURCES` in `set`.

### Create `cpp/moduleName` Folder

### Add line to `freud/__init__.py`

Add `from . import moduleName` so that your module is imported by default.

### Add line to `freud/_freud.pyx`

Add `include "moduleName.pxi"` to `freud/__init__.py`. This must be done to have freud include your python-level code.

### Create `freud/moduleName.pxi` file

This will house the python-level code. If you have a .pxd file exposing C++ classes, make sure to import that:

    cimport freud._moduleName as moduleName

### Create `freud/moduleName.py` file

Make sure there is an import for each C++ class in your module:

    from ._freud import MyC++Class

### Create `freud/_moduleName.pxd`

This file will expose the C++ classes in your module to python.

### Add line to `doc/source/modules.rst`

Make sure your new module is referenced in the documentation.

### Create `doc/source/moduleName.rst`

## Add a new class

To add a new class to an existing module/function (or a newly created one) do the following:

### Create `cpp/moduleName/SubModule.h` and `cpp/moduleName/SubModule.cc`

New classes should be grouped into paired .h, .cc files. There may be a few instances where new classes could be added
to an existing .h, .cc pairing.

### Add line to `freud/moduleName.py` file

Add a line for each C++ class in your module:

    from ._freud import MyC++Class

### Expose C++ class in `freud/_moduleName.pxd`

### Create Python interface in `freud/moduleName.pxi`

You must include sphinx-style documentation.

### Add unit test to `freud/tests`

### Add extra documentation to `doc/source/moduleName.rst`
