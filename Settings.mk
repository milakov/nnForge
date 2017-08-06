BUILD_MODE?=release
ENABLE_CUDA_BACKEND?=yes
ENABLE_CUDA_PROFILING?=no
NNFORGE_USE_NCCL?=no
PROTOBUF_PATH?=
BOOST_PATH?=
OPENCV_PATH?=
CUDNN_PATH?=/usr/local/cuDNN
CUDA_PATH?=/usr/local/cuda
NCCL_PATH?=/usr/local/nccl
NVCC?=nvcc
PROTOC?=protoc
CUDA_FLAGS_ARCH?=-gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=\"sm_60,compute_60\"
NNFORGE_PATH?=../..
NNFORGE_INPUT_DATA_PATH?=~/nnforge/input_data
NNFORGE_WORKING_DATA_PATH?=~/nnforge/working_data

PROTOBUF_LIBS?=-lprotobuf
BOOST_LIBS?=-lboost_thread -lboost_regex -lboost_chrono -lboost_filesystem -lboost_program_options -lboost_random -lboost_system -lboost_date_time
OPENCV_LIBS?=-lopencv_highgui -lopencv_imgproc -lopencv_core
CUDA_LIBS?=-lcudnn -lcurand -lcusparse -lcublas -lcudart
NETCDF_LIBS?=-lnetcdf
MATIO_LIBS?=-lmatio

CPP_HW_ARCHITECTURE?=-march=native # set this to -march=corei7 if you see AVX related errors
CPP_FLAGS_COMMON?=-ffast-math $(CPP_HW_ARCHITECTURE) -mfpmath=sse -msse2 # -mavx
CPP_FLAGS_DEBUG_MODE?=-g
CPP_FLAGS_RELEASE_MODE?=-O3

CPP_FLAGS_OPENMP?=-fopenmp
LD_FLAGS_OPENMP?=-fopenmp

CUDA_FLAGS_COMMON?=-std=c++11 -use_fast_math -DBOOST_NOINLINE='__attribute__ ((noinline))'
CUDA_FLAGS_DEBUG_MODE?=-g -lineinfo
CUDA_FLAGS_RELEASE_MODE?=-O3
