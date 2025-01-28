CC=g++
FLAGS=-std=c++20 -O2 -fopenmp -march=native #-DMKL_LP64 -L${MKLROOT}/lib/intel64/libmkl_sequential.a -lpthread -lm -ldl

#-DMKL_LP64 -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl

#add your own Eigen path
#DIR=-I /mnt/c/lib/Eigen337  
DIR=-I/home/lijun/libs/eigen-3.4.0 #-I"${MKLROOT}/include"

OBJ=main.o sampler.o
HEADERS=mps.hpp mpo.hpp solver.hpp

#Rules....
%.o:%.cpp
	$(CC) -c $(FLAGS) $(DIR) -o $@ $< 

s-mps: $(OBJ)
	$(CC) $(FLAGS) $(OBJ) $(DIR) -o s-mps
	mkdir -p ../bin
	mv s-mps ../bin/

$(OBJ):$(HEADERS)

clean:
	rm -f *.o s-mps 

	
