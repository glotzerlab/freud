#!/bin/bash
echo $(pwd)
if [ -z $1 ]; then
    echo "A platform (\"lin\" or \"mac\" or \"win\") must be provided as the first argument."
    exit 1
fi

TBB_PLATFORM=$1
TBB_VERSION="2021.2.0"
if [[ "${TBB_PLATFORM}" == "win" ]]; then
    TBB_ZIP="oneapi-tbb-${TBB_VERSION}-${TBB_PLATFORM}.zip"
else
    TBB_ZIP="oneapi-tbb-${TBB_VERSION}-${TBB_PLATFORM}.tgz"
fi
curl -L -O "https://github.com/oneapi-src/oneTBB/releases/download/v${TBB_VERSION}/${TBB_ZIP}"
tar -zxvf "${TBB_ZIP}"

# Move to a hard-coded path (defined by CIBW_ENVIRONMENT)
mv "oneapi-tbb-${TBB_VERSION}" /project/tbb
source /project/tbb/env/vars.sh
echo "TBBROOT: ${TBBROOT:-"not found"}"
