USE_BOOST=yes
USE_OPENCV=yes
USE_CUDA=yes

include ../../Settings.mk
ifeq ($(ENABLE_CUDA_BACKEND),yes)
include ../../Main.mk

TARGET=../../lib/libnnforge_$(shell basename `pwd`).a
SOURCES=$(wildcard *.cpp)
OBJECTS=$(SOURCES:%.cpp=%.o)
SOURCES_CUDA=$(wildcard *.cu)
OBJECTS_CUDA=$(SOURCES_CUDA:%.cu=%.o)

all: $(TARGET)

%_cuda_fermi.o : %_cuda_fermi.cu
	$(NVCC) -c $< $(NVCCFLAGS) $(CUDA_FLAGS_ARCH_FERMI) -o $@

%_cuda_kepler.o : %_cuda_kepler.cu
	$(NVCC) -c $< $(NVCCFLAGS) $(CUDA_FLAGS_ARCH_KEPLER) -o $@

%_cuda.o : %_cuda.cu
	$(NVCC) -c $< $(NVCCFLAGS) $(CUDA_FLAGS_ARCH_FERMI) $(CUDA_FLAGS_ARCH_KEPLER) -o $@

$(OBJECTS): $(SOURCES)

$(OBJECTS_CUDA): $(SOURCES_CUDA)

$(TARGET): $(OBJECTS) $(OBJECTS_CUDA)
	$(AR) $(ARFLAGS) $(TARGET) $(OBJECTS) $(OBJECTS_CUDA)

clean:
	$(RM) $(OBJECTS) $(OBJECTS_CUDA) $(TARGET)
else
all:
	@echo 'CUDA backend build is switched off'
clean:

endif

.PHONY: all clean
