#!/bin/bash
git submodule update --init

TBB_VERSION="2021.2.0"
TBB_ZIP="oneapi-tbb-${TBB_VERSION}-lin.tgz"
curl -L -O "https://github.com/oneapi-src/oneTBB/releases/download/v${TBB_VERSION}/${TBB_ZIP}"
tar -zxvf "${TBB_ZIP}"
source "oneapi-tbb-${TBB_VERSION}/env/vars.sh"
echo "TBBROOT: ${TBBROOT:-"not found"}"
