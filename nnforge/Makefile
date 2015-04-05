USE_PROTOBUF=yes
USE_BOOST=yes
USE_OPENCV=yes

include ../Settings.mk
include ../Main.mk

TARGET=../lib/lib$(shell basename `pwd`).a

PROTO_DIR=proto

PROTO_SOURCES=$(wildcard $(PROTO_DIR)/*.proto)
PROTO_GEN_SOURCES=${PROTO_SOURCES:.proto=.pb.cc}
PROTO_GEN_HEADERS=${PROTO_SOURCES:.proto=.pb.h}
PROTO_OBJECTS=${PROTO_GEN_SOURCES:.cc=.o}

SOURCES=$(wildcard *.cpp)
OBJECTS=$(SOURCES:%.cpp=%.o)

$(PROTO_DIR)/%.pb.o: $(PROTO_DIR)/%.pb.cc $(PROTO_DIR)/%.pb.h
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(PROTO_DIR)/%.pb.cc $(PROTO_DIR)/%.pb.h : $(PROTO_DIR)/%.proto
	$(PROTOC) -I=$(PROTO_DIR) --cpp_out=$(PROTO_DIR) $<

all: $(TARGET)

$(OBJECTS): $(SOURCES) $(PROTO_GEN_HEADERS)

$(TARGET): $(OBJECTS) $(PROTO_OBJECTS)
	$(AR) $(ARFLAGS) $(TARGET) $(OBJECTS) $(PROTO_OBJECTS)

clean:
	$(RM) $(OBJECTS) $(PROTO_OBJECTS) $(PROTO_GEN_HEADERS) $(PROTO_GEN_SOURCES) $(TARGET)

.PHONY: all clean
