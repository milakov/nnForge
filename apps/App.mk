APP_NAME=$(shell basename `pwd`)
TARGET=../../bin/$(APP_NAME)
CONFIG_FILE=$(TARGET).cfg
SOURCES+=$(wildcard *.cpp)
OBJECTS+=$(patsubst %.cpp,%.o,$(wildcard *.cpp))


all: $(TARGET) $(CONFIG_FILE)

$(OBJECTS): $(SOURCES)

$(TARGET): $(OBJECTS) $(LDLIBSDEPEND)
	$(CXX) -o $(TARGET) $(OBJECTS) $(LDLIBSDEPEND) $(LDFLAGS)

$(CONFIG_FILE): config.cfg
	$(RM) $(CONFIG_FILE)
	echo 'input_data_folder=$(NNFORGE_INPUT_DATA_PATH)/$(APP_NAME)' >> $(CONFIG_FILE)
	echo 'working_data_folder=$(NNFORGE_WORKING_DATA_PATH)/$(APP_NAME)' >> $(CONFIG_FILE)
	cat config.cfg >> $(CONFIG_FILE)

clean:
	$(RM) $(OBJECTS) $(TARGET) $(CONFIG_FILE)

.PHONY: all clean
