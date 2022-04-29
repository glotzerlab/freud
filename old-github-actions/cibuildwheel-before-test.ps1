$PackageDir = $args[0]
$PyVer = $(python --version)
If ("$PyVer" -Match "Python 3\.6\.") {
    # Python 3.6 is only supported with oldest requirements
    pip install -U -r $PackageDir/.circleci/ci-oldest-reqs.txt --progress-bar=off
} Else If ("$PyVer" -Match "Python 3\.7\.") {
    # Python 3.7 was dropped by NEP 29 so not compatible with some of the newest dependencies
    pip install -U -r $PackageDir/requirements/requirements-test-compatible.txt --progress-bar=off
} Else {
    pip install -U -r $PackageDir/requirements/requirements-test.txt --progress-bar=off
}

# Allow parallel tests to speed up CI
pip install -U pytest-xdist --progress-bar=off
