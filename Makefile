CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

TARGET = test_conv
SRCS = test.cpp conv.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

test.o: test.cpp conv.h
	$(CXX) $(CXXFLAGS) -c test.cpp

conv.o: conv.cpp conv.h
	$(CXX) $(CXXFLAGS) -c conv.cpp

clean:
	rm -f $(OBJS) $(TARGET)