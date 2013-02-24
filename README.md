nnForge
=======

nnForge is a library for training convolutional and fully-connected neural networks. It includes CPU and GPU (CUDA) backends. It is an open-source software distributed under the Apache License v2.0.

Building the library and running the examples:
1) Check Settings.mk file, you might need to make some changes to it:
  a) Define paths to Boost and OpenCV installations nnForge depends on.
  b) Enable or disable CUDA backend - you will need to disable it if you don't have CUDA toolkit installed.
2) Run "./make_all.sh". All the arguments are passed to make, thus you might speed up the build process by specifying -j argument with the number of parallel jobs. This should build the library and examples. 
3) Library files are in lib/ directory once build process successfuly finishes.
4) Examples provided in examples/ directory are built into bin/. Configuration files needed to run those apps are also there.
5) Each example contains its own README with instructions on how to get input data and run the code.

For the time being the easiest way to set up your own project using nnForge library is to copy one of the examples to the subdirectory and modify it for your own needs.
