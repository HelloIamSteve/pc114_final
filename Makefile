# ============================================================
# Makefile for C++ LeNet project
# Current stage:
#   - tensor.cpp      (Tensor4D + OpenCV image loading)
#   - conv.cpp
#   - layers.cpp
#   - weight_loader.cpp
#
# If main.cpp exists, running `make` builds the executable.
# If main.cpp does not exist yet, running `make` compiles core objects.
# ============================================================

CXX = g++

# OpenCV flags for image loading in tensor.cpp.
# Requires OpenCV installed with pkg-config support, usually opencv4.
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null)
OPENCV_LIBS := $(shell pkg-config --libs opencv4 2>/dev/null)

THREAD_FLAGS = -pthread

CXXFLAGS = -std=c++17 -Wall -Wextra -O2 $(OPENCV_CFLAGS) $(THREAD_FLAGS)
LDLIBS = $(OPENCV_LIBS) $(THREAD_FLAGS)

TARGET = main
MAIN_SRC = main.cpp

CORE_SRCS = tensor.cpp conv.cpp layers.cpp weight_loader.cpp model.cpp
CORE_OBJS = $(CORE_SRCS:.cpp=.o)

MAIN_OBJ = $(MAIN_SRC:.cpp=.o)
SRCS = $(CORE_SRCS) $(MAIN_SRC)
OBJS = $(CORE_OBJS) $(MAIN_OBJ)

# If main.cpp exists, build the executable.
# If main.cpp does not exist yet, only compile the core files.
ifneq ($(wildcard $(MAIN_SRC)),)
all: $(TARGET)
else
all: core
endif

core: $(CORE_OBJS)
	@echo "Core objects built successfully. Add main.cpp to build $(TARGET)."

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Header dependencies.
tensor.o: tensor.cpp tensor.h
conv.o: conv.cpp conv.h tensor.h
layers.o: layers.cpp layers.h tensor.h
weight_loader.o: weight_loader.cpp weight_loader.h conv.h layers.h tensor.h
model.o: model.cpp model.h conv.h layers.h tensor.h weight_loader.h
main.o: main.cpp model.h conv.h layers.h tensor.h weight_loader.h

run: $(TARGET)
	./$(TARGET)

debug: CXXFLAGS = -std=c++17 -Wall -Wextra -g -O0 $(OPENCV_CFLAGS)
debug: clean all

# Placeholder targets for future acceleration versions.
openmp:
	@echo "OpenMP target will be added after the serial version is correct."

pthread:
	@echo "Pthread target will be added after the serial version is correct."

cuda:
	@echo "CUDA target will be added after the serial version is correct."

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all core run debug openmp pthread cuda clean