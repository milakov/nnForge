#!/bin/bash -x
mkdir -p lib
mkdir -p bin
cd nnforge
make $@
# cd plain
# make $@
# cd ..
cd cuda
make $@
cd ../..
cd examples
for i in ./*
do
	if [ -d "$i" ];then
		cd $i
		make $@
		cd ..
	fi
done
cd ..
cd apps
for i in ./*
do
	if [ -d "$i" ];then
		cd $i
		make $@
		cd ..
	fi
done
cd ..

