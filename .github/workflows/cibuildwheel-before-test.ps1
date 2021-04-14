$PackageDir = $args[0]
$PyVer = $(python --version)
If ("$PyVer" -Match "Python 3\.6\.") {
    # Python 3.6 is only supported with oldest requirements
    pip install -U -r $PackageDir/.circleci/ci-oldest-reqs.txt --progress-bar=off
} Else {
    pip install -U -r $PackageDir/requirements/requirements-test.txt --progress-bar=off
}
