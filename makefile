# Makefile for MPI C++ program

CXX = mpic++
CXXFLAGS = -std=c++14 -I/opt/homebrew/Cellar/boost/1.84.0_1/include -I/opt/homebrew/Cellar/open-mpi/5.0.1/include
LDFLAGS = -L/opt/homebrew/Cellar/boost/1.84.0_1/lib -L/opt/homebrew/Cellar/open-mpi/5.0.1/lib
LDLIBS = -lboost_serialization -lmpi

# List of source files
SRCS = avoa.cpp main.cpp

# List of object files
OBJS = $(SRCS:.cpp=.o)

# Executable name
EXEC = main

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(EXEC)
