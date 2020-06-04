import unittest
import doctest
import freud
import inspect


def load_tests(loader, tests, ignore):
    for name, member in inspect.getmembers(freud):
        if inspect.ismodule(member):
            tests.addTests(doctest.DocTestSuite(member))
    return tests


if __name__ == '__main__':
    unittest.main()
