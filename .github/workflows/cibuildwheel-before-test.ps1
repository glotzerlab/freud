$PACKAGE_DIR = $args[0]
$PYTHON_VERSION = $(python --version)

pip install -U -r ${PACKAGE_DIR}/requirements/requirements-test.txt --progress-bar=off

# Allow parallel tests to speed up CI
#pip install -U pytest-xdist --progress-bar=off
