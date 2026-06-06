# ============================================================
# Makefile for C++ LeNet project
# Current stage:
#   - tensor.cpp
#   - conv.cpp
#   - layers.cpp
#
# Since test.cpp / main() is not created yet, running `make`
# will currently compile only the core object files.
# After you add test.cpp, running `make` will build the executable.
# ============================================================

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

TARGET = lenet_test
MAIN_SRC = test.cpp

CORE_SRCS = tensor.cpp conv.cpp layers.cpp
CORE_OBJS = $(CORE_SRCS:.cpp=.o)

MAIN_OBJ = $(MAIN_SRC:.cpp=.o)
SRCS = $(CORE_SRCS) $(MAIN_SRC)
OBJS = $(CORE_OBJS) $(MAIN_OBJ)

# If test.cpp exists, build the executable.
# If test.cpp does not exist yet, only compile the core files.
ifneq ($(wildcard $(MAIN_SRC)),)
all: $(TARGET)
else
all: core
endif

# Compile core object files only.
core: $(CORE_OBJS)
	@echo "Core objects built successfully. Add test.cpp to build $(TARGET)."

# Build executable after test.cpp exists.
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Generic C++ compile rule.
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Header dependencies.
tensor.o: tensor.cpp tensor.h
conv.o: conv.cpp conv.h tensor.h
layers.o: layers.cpp layers.h tensor.h
test.o: test.cpp conv.h layers.h tensor.h

# Run the executable.
run: $(TARGET)
	./$(TARGET)

# Debug build.
debug: CXXFLAGS = -std=c++17 -Wall -Wextra -g -O0
debug: clean all

# Placeholder targets for future acceleration versions.
# We will activate these after serial LeNet inference is correct.
openmp:
	@echo "OpenMP target will be added after the serial version is correct."

pthread:
	@echo "Pthread target will be added after the serial version is correct."

cuda:
	@echo "CUDA target will be added after the serial version is correct."

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all core run debug openmp pthread cuda clean