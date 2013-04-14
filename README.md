nnForge
=======

nnForge is a library for training convolutional and fully-connected neural networks. It includes CPU and GPU (CUDA) backends.
It is an open-source software distributed under the [Apache License v2.0](http://www.apache.org/licenses/LICENSE-2.0).

Authors
-------
nnForge is designed and implemented by [Maxim Milakov](http://milakov.org).

Build
-----

1. Check Settings.mk file, you might need to make some changes to it:
	* Define paths to [Boost](http://www.boost.org/) and [OpenCV](http://opencv.org/) installations nnForge depends on.
	* Enable or disable CUDA backend - you will need to disable it if you don't have [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) installed.
2. Run "./make_all.sh". This should build the library and the examples. All the arguments are passed to make, thus you might speed up the build process by specifying -j argument with the number of parallel jobs. The build process might take about 15 minutes on a modern desktop.
3. Library files are in lib/ directory once build process successfuly finishes.
4. Examples provided in examples/ directory are built into bin/. Configuration files needed to run those apps are also there.

Run examples
------------

Each example contains its own README with instructions on how to get input data and run the code.

Setup your project
------------------

For the time being the easiest way to set up your own project using nnForge library is to copy one of the examples to the subdirectory and modify it for your own needs.
