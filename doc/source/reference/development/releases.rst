=========================
Making **freud** Releases
=========================

Release Process
===============

Documented below are the steps needed to make a new release of **freud**.

1. Create a release branch, numbered according to `Sematic Versioning <https://semver.org/spec/v2.0.0.html>`_:

.. code-block:: bash

   git checkout -b release/vX.Y.Z

Changelog
---------

2. Review headings (Added, Changed, Fixed, Deprecated, Removed) and ensure consistent formatting.
3. Update the release version and release date from ``next`` to ``vX.Y.Z - YYYY-MM-DD``.

Submodules
----------

4. Update git submodules (optional, but should be done regularly).

Code Formatting
---------------

5. Reformat C++ code with ``clang-format`` 6.0:

.. code-block:: bash

   clang-format -style=file cpp/**/*

Generate Cythonized C++
-----------------------

6. Rebuild the project with ``--ENABLE-CYTHON`` so that the files in ``freud/*.cpp`` are updated.
   Ideally, do this on a system with a relatively clean ``PATH`` and a system-provided Python, or else some user paths (including usernames) may be included in the result.

Contributors
------------

7. Update the contributor list:

.. code-block:: bash

   git shortlog -sne > contributors.txt

Bump version
------------

8. Commit previous changes before running ``bumpversion``.
9. Use the `bumpversion package <https://pypi.org/project/bumpversion/>`_ to increase the version number and automatically generate a git tag:

.. code-block:: bash

   bumpversion patch  # for X.Y.Z
   bumpversion minor  # for X.Y
   bumpversion major  # for X

10. Push the release branch to the remote:

.. code-block:: bash

    git push -u origin release/vX.Y.Z

11. Ensure that ReadTheDocs and continuous integration pass (you will need to manually enable the branch on ReadTheDocs' web interface to test it).
    Then push the tag:

.. code-block:: bash

    git push --tags

Automatic Builds
----------------

12. Pushing the tag will cause CircleCI to create a release for PyPI automatically (see automation in ``.circleci/config.yml``). Make sure this succeeds -- it takes a while to run.

13. Create a pull request and merge the release branch into the ``master`` branch. Delete the release branch on ReadTheDocs' web interface, since there is now a tagged version.

14. The conda-forge autotick bot should discover that the PyPI source distribution has changed, and will create a pull request to the `conda-forge feedstock <https://github.com/conda-forge/freud-feedstock/>`_.
    This pull request may take a few hours to appear.
    If other changes are needed in the conda-forge recipe (e.g. new dependencies), follow the conda-forge documentation to create a pull request from *your own fork* of the feedstock.

Release Announcement
--------------------

15. Verify that ReadTheDocs, PyPI, and conda-forge have been updated to the newest version.

16. Send a release notification via the `freud-users group <https://groups.google.com/forum/#!forum/freud-users>`_.
    Follow the template of previous release notifications.
