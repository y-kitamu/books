#!/bin/bash

# set -u # 定義されていない変数があったら停止

cur_dir=$(pwd)
src_dir=$(dirname ${BASH_SOURCE:-$0})
build_dir=${src_dir}/../edk2

cd ${build_dir} && source edksetup.sh && build && cd ${cur_dir}

target=${build_dir}/Build/MikanLoaderX64/DEBUG_CLANG38/X64/Loader.efi
