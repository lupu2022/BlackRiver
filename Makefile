.PHONY: all
all: br 

FLAGS = -Wall 
INC = -I. -I/engine/includes \
	  -I/home/teaonly/opt/nccl/include \
	  -I/home/teaonly/opt/msgpack \
	  -I/usr/local/cuda/include \
	  -I/usr/lib/x86_64-linux-gnu/openmpi/include \
	  -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi 

LINK = -L/home/teaonly/opt/nccl/lib -lnccl \
	   -L/usr/local/cuda/lib64 -lcudnn -lcudart -lcublas -lcublasLt \
	   -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi -lmpi_cxx

ENGINE_LIBS = kernels/build/libkernels.a

OBJS_DIR = ./objs
BR_SRCS = br.cpp computing.cpp tensortype.cpp cuda_tensor.cpp dag.cpp 
BR_OBJS = ${BR_SRCS:%.cpp=$(OBJS_DIR)/%.o}

$(OBJS_DIR)/%.o : %.cpp
	@mkdir -p $(@D)
	g++ $(FLAGS) $(INC) -c -o $@ $< 

br: $(BR_OBJS) 
	g++ $(FLAGS) -o $@ $(BR_OBJS) $(ENGINE_LIBS) $(LINK)

clean:
	rm -rf $(OBJS_DIR)
	rm -f br

