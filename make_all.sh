#!/bin/bash -x
mkdir -p lib
mkdir -p bin
cd nnforge
make $@
cd plain
make $@
cd ../cuda
make $@
cd ../../examples
cd gtsrb
make $@
cd ../..

