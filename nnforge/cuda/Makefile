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

%.o : %.cu
	$(NVCC) -c $< $(NVCCFLAGS) $(CUDA_FLAGS_ARCH) -o $@

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
