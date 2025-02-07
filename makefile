CXX = mpic++
CXXFLAGS = -std=c++14 -I/opt/homebrew/Cellar/boost/1.87.0/include -I/opt/homebrew/Cellar/open-mpi/5.0.1/include
LDFLAGS = -L/opt/homebrew/Cellar/boost/1.87.0/lib -L/opt/homebrew/Cellar/open-mpi/5.0.1/lib
LDLIBS = -lboost_serialization -lmpi

SRCS = avoa.cpp main.cpp
OBJS = $(SRCS:.cpp=.o)
EXEC = main
NUM_PROCESSES = 4

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

run: $(EXEC)
	mpiexec -n $(NUM_PROCESSES) ./$(EXEC)

clean:
	rm -f $(OBJS) $(EXEC)
