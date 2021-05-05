import doctest
import inspect

import freud


def load_tests(loader, tests, ignore):
    optionflags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    for name, member in inspect.getmembers(freud):
        if inspect.ismodule(member):
            tests.addTests(doctest.DocTestSuite(member, optionflags=optionflags))
    return tests
