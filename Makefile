CC=nvcc
CUFLAGS= -w -O3 -gencode arch=compute_75,code=compute_75 -lineinfo
SOURCES= main.cu mf_methods.cu model_init.cu
INC = -I . -I ./mascot -I ./afp -I ./muppet -I ./mpt -I ./sgd
LIBS = -lboost_system -lboost_filesystem
EXECUTABLE=quantized_mf
OBJECTS=$(SOURCES:.cpp=.o)
	OBJECTS=$(patsubst %.cpp,%.o,$(patsubst %.cu,%.o,$(SOURCES)))
	DEPS=mf_methods.h io_utils.h preprocess_utils.h common.h common_struct.h model_init.h rmse.h precision_switching.h mascot_sgd_kernel_k64.h mascot_sgd_kernel.h ./afp/afp_sgd_kernel.h ./afp/afp_sgd_kernel_k64.h ./muppet/muppet_sgd_kernel.h ./muppet/muppet_sgd_kernel_k64.h ./mpt/mpt_sgd_kernel.h ./mpt/mpt_sgd_kernel_k64.h reduce_kernel.h ./sgd/sgd_kernel.h ./sgd/sgd_kernel_k64.h
	VPATH= ./afp ./mascot ./muppet ./mpt ./sgd
	DATA_PATH=

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	        $(CC) $(CUFLAGS)  $^ -o $@ $(INC) $(LIBS)

%.o: %.cu $(DEPS)
	        $(CC) -c $< -o $@ $(CUFLAGS) $(INC)

# %.o: %.cpp $(DEPS)
# 	        $(CC) -c $< -o $@ $(CUFLAGS) $(INC)

clean:
	        rm ./quantized_mf *.o
test:
	./quantized_mf -i $(DATA_PATH)/ML25M/u1.base -y $(DATA_PATH)/ML25M/u1.test -o trained_model/mf_parameter_mascot_ML25M -wg 2048 -bl 128 -k 128 -l 50 -a 0.01 -d 0.1 -ug 100 -ig 100 -e 20 -s 0.05 -it 2 -v 1
