Contributions are welcomed via pull requests. First, contact the _freud_ developers prior to beginning
your work to ensure that your plans mesh well with the planned development direction and standards set for the project.
Then implement your code.

Submit a pull request. Multiple developers and/or users will review requested changes and make comments.
The lead developer(s) will merge into the `master` branch after the review is complete and approved.

# Features

## Implement functionality in a general and flexible fashion

The _freud_ library provides a lot of flexibility to the user. Your pull request should provide something that is
applicable to a variety of use-cases and not just the one thing you might need it to do for your research. Speak to the
lead developers before writing your code, and they will help you make design choices that allow flexibility.

## Do not degrade performance of existing code paths

New functionalities should only activate expensive code paths when they are requested by the user.
Do not slow down existing code.

## Add dependencies only if absolutely necessary

In order to make _freud_ as widely available as possible, we try to keep the number of dependencies to a minimum.
If you need a feature present in an external library, follow the following steps:

1. Add to _freud_ itself if it's simple or if other modules would benefit:
    * Example: Added simple tensor math for CubaticOrderParameter
2. Add via submodule if the code exists externally:
    * Example: _fsph_
3. Contact _freud_ developers to inquire if the library you'd like as a dependency fits in with the overall design/goals
of _freud_.

# Version control

## Base your work off the correct branch

Bug fixes should be based on `maint`. New features should be based on `master`.

## Propose a single set of related changes

Changes proposed in a single topic branch / pull request should all be related to each other. Don't propose too
many changes at once, review becomes challenging. Multiple new features that are loosely coupled should be completed
in separate topic branches. It is OK if the branch for `feature2` is based on `feature1` - as long as it is made clear
that `feature1` should be merged before the review of `feature2`. It is better to merge both `feature1` and `feature2`
into a temporary integration branch during testing.

## Keep changes to a minimum

Don't go and "fix" spelling errors all over the code, or make lots of whitespace changes along with a new feature.
If there are spelling errors to fix, propose that in a separate pull request :)

## Agree to the contributor agreement

All contributors must agree to the Contributor Agreement ([ContributorAgreement.md](ContributorAgreement.md))
before their pull request can be merged.

# Source code

## Use a consistent style

It is important to have a consistent style throughout the source code.
Follow the source conventions defined in the documentation for all code.

## Document code with comments

Use doxygen header comments for classes, functions, etc.
Also comment complex sections of code so that other developers can understand them.

## Compiles without warnings

Your changes should compile without warnings.

# Tests

## Write unit tests

All new functionality in _freud_ should be tested with automatic unit tests that execute in a few seconds (if your
specific test requires a long amount of time, please alert the _freud_ developers as to why this is required so that
your test can be opted-out of for "regular" unit-testing). High level features should be tested from Python, and the
Python tests should attempt to cover all options that the user can select.

## Validity tests

In addition to the unit tests, the developer should run research-scale analysis using the new functionality and
ensure that it behaves as intended.

# User documentation

## Write user documentation

User documentation for the user facing script commands should be documented with docstrings in [Google format](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
Include examples on using new functionality.

## Document version status

Each user-facing python class, method, etc... with a docstring should have [versionadded, versionchanged, and
deprecated sphinx paragraphs](https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directive-versionadded) so that users will be aware of
how functionality changes from version to version.

## Add developer to the credits

Developers need to be credited for their work. Update the [credits documentation](doc/source/credits.rst)
to reference what each developer has contributed to the code.

## Update Change Log

Add a short concise entry describing the change to the [ChangeLog.md](ChangeLog.md).
