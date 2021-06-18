CC=nvcc
CUFLAGS= -w -O3 -gencode arch=compute_75,code=compute_75 -lineinfo
SOURCES= test.cu
INC = -I . -I ..
LIBS = -lboost_system -lboost_filesystem
EXECUTABLE=test_mf
OBJECTS=$(SOURCES:.cpp=.o)
	OBJECTS=$(patsubst %.cpp,%.o,$(patsubst %.cu,%.o,$(SOURCES)))
	DATA_PATH=/home/winston/projects/new_vscodefiles/CUDA_SVD/data
	DEPS= ../io_utils.h 

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	        $(CC) $(CUFLAGS)  $^ -o $@ $(INC) $(LIBS)

%.o: %.cu $(DEPS)
	        $(CC) -c $< -o $@ $(CUFLAGS) $(INC)

clean:
	        rm ./test_mf *.o
test:
	#./test_mf -i ../trained_model/mf_parameter_fp32_ML10M.txt -y $(DATA_PATH)/ML10M/reconst_u1.test

	#./test_mf -i ../trained_model/mf_parameter_fp32_ML10M.txt -y $(DATA_PATH)/ML10M/u1.test
	#./test_mf -i ../trained_model/mf_parameter_fp32_ML25M.txt -y $(DATA_PATH)/ML25M/u1.test

	#./test_mf -i ../trained_model/mf_parameter_mascot_ML25M_reconst.txt -y $(DATA_PATH)/ML25M/reconst_u1.test -v 1
	./test_mf -i ../trained_model/mf_parameter_mascot_yahoo_reconst.txt -y $(DATA_PATH)/Yahoo_music/reconst_validationIdx1.txt -v 1

	#./test_mf -i ../trained_model/mf_parameter_fp32_Netflix.txt -y $(DATA_PATH)/netflix/netflix_cumf_test.tsv
	#./test_mf -i ../trained_model/mf_parameter_fp32_yahoo.txt -y $(DATA_PATH)/Yahoo_music/validationIdx1.txt
	#./test_mf -i ../trained_model/ML10M_mf_parameter_mascot_ML25M.txt -y $(DATA_PATH)/ML25M/reconst_u1.test
