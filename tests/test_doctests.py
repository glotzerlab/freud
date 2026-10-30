# Copyright (c) 2010-2026 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import doctest
import inspect

import pytest

import freud

try:
    import spatula
except ImportError:
    spatula = None

# Docstrings that require optional dependencies to execute.
_OPTIONAL_DEPENDENCY_DOCTESTS = () if spatula else ("freud.environment.PGOP",)


def fetch_doctests():
    finder = doctest.DocTestFinder()
    for _name, member in inspect.getmembers(freud):
        if inspect.ismodule(member):
            for docstring in finder.find(member):
                if docstring.examples and docstring.name not in (
                    _OPTIONAL_DEPENDENCY_DOCTESTS
                ):
                    yield docstring


class TestDoctests:
    @pytest.mark.parametrize(
        "docstring", fetch_doctests(), ids=lambda docstring: docstring.name
    )
    def test_docstring(self, docstring):
        optionflags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        runner = doctest.DocTestRunner(optionflags=optionflags)
        runner.run(docstring)
        results = runner.summarize()
        if results.failed:
            raise AssertionError(results)
