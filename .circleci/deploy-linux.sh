#!/bin/bash

set -e

# PYPI_USERNAME - (Required) Username for the publisher's account on PyPI
# PYPI_PASSWORD - (Required, Secret) Password for the publisher's account on PyPI

cat << EOF > ~/.pypirc
[distutils]
index-servers=
    pypi
    testpypi

[pypi]
username: ${PYPI_USERNAME}
password: ${PYPI_PASSWORD}

[testpypi]
repository: https://test.pypi.org/legacy/
username: ${PYPI_USERNAME}
password: ${PYPI_PASSWORD}
EOF

if [ -z $1 ]; then
    echo "A repository (\"pypi\" or \"testpypi\") must be provided as the first argument."
    exit 1
fi

# Build TBB
cd ~/
git clone https://github.com/oneapi-src/oneTBB.git
cd oneTBB
mkdir -p build
cd build
cmake ../ -DTBB_TEST=OFF
make
cmake -DCOMPONENT=runtime -P cmake_install.cmake
cmake -DCOMPONENT=devel -P cmake_install.cmake
BUILD_DIR=$(dirname $(find -name vars.sh))
cd ${BUILD_DIR}
source vars.sh
cd ~/

# Build wheels for Python 3.6, 3.7, 3.8, 3.9
PYBINS="/opt/python/cp3[6-9]*/bin"

for PYBIN in $PYBINS; do
  echo "Building for $(${PYBIN}/python --version)"

  # Need to export the current bin path so that scikit-build can find the pip
  # installed cmake binary.
  export PATH=${PYBIN}:${PATH}
  "${PYBIN}/python" -m pip install cython scikit-build cmake --ignore-installed -q --progress-bar=off
  rm -rf numpy-1.14.6
  curl -sSLO https://github.com/numpy/numpy/archive/v1.14.6.tar.gz
  tar -xzf v1.14.6.tar.gz
  cd numpy-1.14.6
  rm -f numpy/random/mtrand/mtrand.c
  rm -f PKG-INFO
  "${PYBIN}/python" -m pip install . --no-deps --ignore-installed -v --progress-bar=off -q
  "${PYBIN}/pip" wheel ~/ci/freud/ -w ~/wheelhouse/ --no-deps --no-build-isolation --no-use-pep517
done

# Install patched auditwheel (fixes RPATHs for libfreud/libtbb, issue #136).
cd ~/
git clone https://github.com/mayeut/auditwheel.git -b issue136 auditwheel
cd auditwheel
/opt/_internal/tools/bin/pip install -e .

# Update RPATH for wheels
for whl in ~/wheelhouse/freud*.whl; do
  auditwheel repair "$whl" -w ~/ci/freud/wheelhouse/
done

# Install from and test all wheels
for PYBIN in $PYBINS; do
  echo "Testing for $(${PYBIN}/python --version)"

  "${PYBIN}/python" -m pip install freud_analysis --no-deps --no-index -f ~/ci/freud/wheelhouse
  if [[ $("${PYBIN}/python" --version 2>&1) == *"3.6."* ]]; then
    # Python 3.6 is only supported with oldest requirements
    "${PYBIN}/python" -m pip install -U -r ~/ci/freud/.circleci/ci-oldest-reqs.txt --progress-bar=off
  else
    "${PYBIN}/python" -m pip install -U -r ~/ci/freud/requirements/requirements-test.txt --progress-bar=off
  fi
  cd ~/ci/freud/tests/
  "${PYBIN}/python" -m pytest . -v
done

# Build source distribution using whichever Python appears last
cd ..
"${PYBIN}/python" setup.py sdist --dist-dir ~/ci/freud/wheelhouse/

"${PYBIN}/pip" install --user twine

"${PYBIN}/python" -m twine upload --skip-existing --repository $1 ~/ci/freud/wheelhouse/*
