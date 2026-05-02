CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

TARGET = test_conv
SRCS = test.cpp conv.cpp tensor.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

test.o: test.cpp conv.o tensor.o
	$(CXX) $(CXXFLAGS) -c test.cpp

conv.o: conv.cpp conv.h tensor.o
	$(CXX) $(CXXFLAGS) -c conv.cpp

tensor.o: tensor.cpp tensor.h
	$(CXX) $(CXXFLAGS) -c tensor.cpp

clean:
	rm -f $(OBJS) $(TARGET)