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
NVCC = nvcc

# OpenCV flags for image loading in tensor.cpp.
# Requires OpenCV installed with pkg-config support, usually opencv4.
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null)
OPENCV_LIBS := $(shell pkg-config --libs opencv4 2>/dev/null)

THREAD_FLAGS = -pthread
OPENMP_FLAGS = -fopenmp

CXXFLAGS = -std=c++17 -Wall -Wextra $(OPENCV_CFLAGS) $(THREAD_FLAGS) $(OPENMP_FLAGS)
NVCCFLAGS = -std=c++17 $(OPENCV_CFLAGS) \
            -Xcompiler "-Wall -Wextra $(THREAD_FLAGS) $(OPENMP_FLAGS)"

LINKFLAGS = $(OPENCV_LIBS) -Xcompiler "$(THREAD_FLAGS) $(OPENMP_FLAGS)"

TARGET = main
MAIN_SRC = main.cu

CPP_SRCS = tensor.cpp conv.cpp layers.cpp weight_loader.cpp
CPP_OBJS = $(CPP_SRCS:.cpp=.o)

CU_SRCS = model.cu
CU_OBJS = $(CU_SRCS:.cu=.o)

CORE_OBJS = $(CPP_OBJS) $(CU_OBJS)

MAIN_OBJ = $(MAIN_SRC:.cu=.o)
SRCS = $(CPP_SRCS) $(CU_SRCS) $(MAIN_SRC)
OBJS = $(CPP_OBJS) $(CU_OBJS) $(MAIN_OBJ)

# If main.cu exists, build the executable.
# If main.cu does not exist yet, only compile the core files.
ifneq ($(wildcard $(MAIN_SRC)),)
all: $(TARGET)
else
all: core
endif

core: $(CORE_OBJS)
	@echo "Core objects built successfully. Add main.cpp to build $(TARGET)."

$(TARGET): $(OBJS)
	$(NVCC) -o $@ $^ $(NVCCFLAGS) $(LINKFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Header dependencies.
tensor.o: tensor.cpp tensor.h
conv.o: conv.cpp conv.h tensor.h
layers.o: layers.cpp layers.h tensor.h
weight_loader.o: weight_loader.cpp weight_loader.h conv.h layers.h tensor.h
model.o: model.cu model.h conv.h layers.h tensor.h weight_loader.h
main.o: main.cu model.h conv.h layers.h tensor.h weight_loader.h

run: $(TARGET)
	./$(TARGET)

debug: CXXFLAGS = -std=c++17 -Wall -Wextra -g -O0 $(OPENCV_CFLAGS) $(THREAD_FLAGS) $(OPENMP_FLAGS)
debug: NVCCFLAGS = -std=c++17 -g -G $(OPENCV_CFLAGS) -Xcompiler "-Wall -Wextra -g -O0 $(THREAD_FLAGS) $(OPENMP_FLAGS)"
debug: clean all

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all core run debug openmp pthread cuda clean