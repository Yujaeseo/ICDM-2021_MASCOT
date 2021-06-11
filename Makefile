CC=nvcc
#CUFLAGS= -w -gencode arch=compute_35,code=compute_35 -lineinfo -Xptxas=-O3 \
	         -gencode arch=compute_37,code=compute_37 \
		          -gencode arch=compute_50,code=compute_50 \
			           -gencode arch=compute_75,code=compute_75
#CUFLAGS = -w -O3 -gencode arch=compute_75,code=compute_75 -lineinfo -DUSER_GROUP_NUM=$(USER_GROUP_NUM_INP) -DITEM_GROUP_NUM=$(ITEM_GROUP_NUM_INP)
CUFLAGS = -w -O3 -gencode arch=compute_75,code=compute_75 -lineinfo
#CUFLAGS = -w  -Xptxas -dlcm=ca
SOURCES= main.cpp model_init.cu sgd.cu
INC = -I .
LIBS = -lboost_thread -lboost_system -lboost_iostreams -lutil -lboost_filesystem
EXECUTABLE=quantized_svd
OBJECTS=$(SOURCES:.cpp=.o)
	OBJECTS=$(patsubst %.cpp,%.o,$(patsubst %.cu,%.o,$(SOURCES)))
	DEPS= common_struct.h io_utils.h model_init.h sgd.h preprocess_utils.h sgd_kernel_128.h rmse.h precision_switching.h test_kernel.h statistics.h fixed_point_sgd_kernel_128.h fixed_point_sgd_kernel_128_cvpr.h sgd_kernel_64.h fixed_point_sgd_kernel_64.h fixed_point_sgd_kernel_64_cvpr.h
	VPATH=
	DATA_PATH=

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	        $(CC) $(CUFLAGS)  $^ -o $@ $(INC) $(LIBS)

%.o: %.cu $(DEPS)
	        $(CC) -c $< -o $@ $(CUFLAGS) $(INC)

%.o: %.cpp $(DEPS)
	        $(CC) -c $< -o $@ $(CUFLAGS) $(INC)

clean:
	        rm ./quantized_svd *.o
test:
	./quantized_svd -i $(DATA_PATH)/ML10M/u1.base -y $(DATA_PATH)/ML10M/u1.test -o testfp16_grad_diversity -k 128 -wg 2048 -bl 128 -l 50 -ug 100 -ig 100 -d 0.1 -a 0.01 -v 24 -g 4 -s 0.05 -e -0.05 -it 1
	./quantized_svd -i $(DATA_PATH)/ML10M/u1.base -y $(DATA_PATH)/ML10M/u1.test -o testfp16_grad_diversity -k 128 -wg 2048 -bl 128 -l 50 -ug 100 -ig 100 -d 0.1 -a 0.01 -v 12 -g 4 -s 0.05 -e -0.05 -it 1
