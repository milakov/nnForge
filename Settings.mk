BUILD_MODE=release
ENABLE_CUDA_BACKEND=yes
ENABLE_CUDA_PROFILING=no
CPP11COMPILER=no
BOOST_PATH=/usr/local
OPENCV_PATH=/usr/local
NETCDF_INSTALLED=yes
NETCDF_PATH=
CUDA_PATH=/usr/local/cuda
NVCC=nvcc
NNFORGE_PATH=../..
NNFORGE_INPUT_DATA_PATH=/home/max/nnforge/input_data
NNFORGE_WORKING_DATA_PATH=/home/max/nnforge/working_data

BOOST_LIBS=-lboost_thread-mt -lboost_regex-mt -lboost_chrono-mt -lboost_filesystem-mt -lboost_program_options-mt -lboost_random-mt -lboost_system-mt -lboost_date_time-mt
OPENCV_LIBS=-lopencv_highgui -lopencv_imgproc -lopencv_core
NETCDF_LIBS=-lnetcdf

CPP_FLAGS_CPP11=-std=c++11
CPP_HW_ARCHITECTURE=-march=native
CPP_FLAGS_COMMON=-ffast-math $(CPP_HW_ARCHITECTURE) -mfpmath=sse -msse2 # -mavx
CPP_FLAGS_DEBUG_MODE=-g
CPP_FLAGS_RELEASE_MODE=-O3

CPP_FLAGS_OPENMP=-fopenmp
LD_FLAGS_OPENMP=-fopenmp

CUDA_FLAGS_COMMON=-use_fast_math -DBOOST_NOINLINE='__attribute__ ((noinline))'
CUDA_FLAGS_ARCH_FERMI=-gencode=arch=compute_20,code=sm_20
CUDA_FLAGS_ARCH_KEPLER=-gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=\"sm_35,compute_35\"
CUDA_FLAGS_DEBUG_MODE=-g -lineinfo
CUDA_FLAGS_RELEASE_MODE=-O3
