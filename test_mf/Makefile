CC=nvcc
CUFLAGS= -w -O3 -gencode arch=compute_75,code=compute_75 -lineinfo
SOURCES= test.cu
INC = -I . -I ..
LIBS = -lboost_system -lboost_filesystem
EXECUTABLE=test_mf
OBJECTS=$(SOURCES:.cpp=.o)
	OBJECTS=$(patsubst %.cpp,%.o,$(patsubst %.cu,%.o,$(SOURCES)))
	DATA_PATH=
	DEPS= ../io_utils.h 

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	        $(CC) $(CUFLAGS)  $^ -o $@ $(INC) $(LIBS)

%.o: %.cu $(DEPS)
	        $(CC) -c $< -o $@ $(CUFLAGS) $(INC)

clean:
	        rm ./test_mf *.o
test:
	./test_mf -i ../trained_model/mf_parameter_mascot_ML25M.txt -y $(DATA_PATH)/ML25M/reconst_u1.test -v 1
