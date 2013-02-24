USE_BOOST=yes
USE_OPENCV=yes
USE_OPENMP=yes

include ../../Settings.mk
include ../../Main.mk

TARGET=../../lib/libnnforge_$(shell basename `pwd`).a
SOURCES=$(wildcard *.cpp)
OBJECTS=$(SOURCES:%.cpp=%.o)

all: $(TARGET)

$(OBJECTS): $(SOURCES)

$(TARGET): $(OBJECTS)
	$(AR) $(ARFLAGS) $(TARGET) $(OBJECTS)

clean:
	$(RM) $(OBJECTS) $(TARGET)

.PHONY: all clean
