#!/bin/bash

cur_dir=$(pwd)

kernel_dir=$(dirname ${BASH_SOURCE:-$0})/kernel
cd ${kernel_dir} && make
cd ${cur_dir}
