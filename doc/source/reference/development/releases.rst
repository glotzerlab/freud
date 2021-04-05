=========================
Making **freud** Releases
=========================

Release Process
===============

Documented below are the steps needed to make a new release of **freud**.

Starting the Release
--------------------

- Review open pull requests, issues, and milestones.
  Before making a release, every issue or pull request assigned to that release version should be completed or moved to a later milestone.
  Additionally, all resolved issues and merged pull requests since the last release should be assigned to the current milestone.
  Some exceptions can be made for simplicity, e.g. automated updates from Dependabot don't need to be assigned to the milestone.

- Create a release branch, numbered according to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`__:

.. code-block:: bash

    git checkout -b release/vX.Y.Z

Changelog
---------

- Review headings (Added, Changed, Fixed, Deprecated, Removed) and ensure consistent formatting.
- Update the release version and release date from ``next`` to ``vX.Y.Z - YYYY-MM-DD``.

Dependencies
------------

- Make sure that package requirements are consistent in ``setup.py`` and ``requirements.txt``.

- Review Python version support (e.g. adding tests for new versions, dropping old versions no sooner than allowed by `NEP 29 <https://numpy.org/neps/nep-0029-deprecation_policy.html>`__).

- Update git submodules. Example for the ``freud-examples`` submodule:

.. code-block:: bash

    # Update freud-examples submodule:
    cd doc/source/gettingstarted/examples
    git pull
    git checkout master
    # Then go to the repository root and commit the changes.

Code Formatting
---------------

- Update pre-commit hooks:

.. code-block:: bash

    pre-commit autoupdate

- If necessary, pre-commit hooks can be run manually to apply code formatting.
  This is typically handled by CI (which requires pre-commit hooks to succeed), but updated hooks may require a re-run.

.. code-block:: bash

    # Apply default pre-commit hooks (Python and generic formatters)
    pre-commit run --all-files

    # Apply C++ pre-commit hooks (requires clang-format and other tools)
    pre-commit run --all-files -c .pre-commit-config-cpp.yaml

Contributors
------------

- Update the contributor list:

.. code-block:: bash

    git shortlog -sne > contributors.txt

Bump version
------------

- Commit previous changes before running ``bumpversion``.
- Use the `bumpversion package <https://pypi.org/project/bumpversion/>`_ to increase the version number and automatically generate a git tag:

.. code-block:: bash

    bumpversion patch  # for X.Y.Z
    bumpversion minor  # for X.Y
    bumpversion major  # for X

- Push the release branch to the remote:

.. code-block:: bash

    git push -u origin release/vX.Y.Z

- Create a pull request for that branch.

- Ensure that ReadTheDocs and continuous integration pass on the release branch's pull request.
  Pushing the release branch will cause CircleCI to create a release for TestPyPI automatically (see automation in ``.circleci/config.yml``).
  Make sure this succeeds -- it takes a while to run.
  Review the `TestPyPI builds <https://test.pypi.org/project/freud-analysis/>`__ to ensure the README looks correct.
  Then push the tag:

.. code-block:: bash

    git push --tags

Automatic Builds
----------------

- Pushing the tag will cause CircleCI to create a release for PyPI automatically (see automation in ``.circleci/config.yml``).
  Make sure this succeeds -- it takes a while to run.

- Merge the release branch pull request into the ``master`` branch.

- The conda-forge autotick bot should discover that the PyPI source distribution has changed, and will create a pull request to the `conda-forge feedstock <https://github.com/conda-forge/freud-feedstock/>`_.
  This pull request may take a few hours to appear.
  If other changes are needed in the conda-forge recipe (e.g. new dependencies), follow the conda-forge documentation to create a pull request from *your own fork* of the feedstock.
  Merge the pull request after all continuous integration passes to trigger release builds for conda-forge.

Release Announcement
--------------------

- Verify that ReadTheDocs, PyPI, and conda-forge have been updated to the newest version.

- Make a GitHub release from the `tag on GitHub <https://github.com/glotzerlab/freud/tags>`__ and clicking "Create release."
  Follow the template from previous release notifications.

- Send a release notification via the `freud-users group <https://groups.google.com/forum/#!forum/freud-users>`__.
  Follow the template from previous release notifications.
