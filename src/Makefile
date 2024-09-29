CC=icpx
FLAGS=-std=c++20 -O2 -qopenmp -xCORE-AVX512 -m64 #-DMKL_LP64 -L${MKLROOT}/lib/intel64/libmkl_sequential.a -lpthread -lm -ldl

#-DMKL_LP64 -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl

#add your own Eigen path
#DIR=-I /mnt/c/lib/Eigen337  
DIR=-I/home/raid/lw755/lib/Eigen340/eigen-3.4.0/ #-I"${MKLROOT}/include"

OBJ=main.o sampler.o
HEADERS=mps.hpp mpo.hpp solver.hpp

#Rules....
%.o:%.cpp
	$(CC) -c $(FLAGS) $(DIR) -o $@ $< 

qc-dmrg: $(OBJ)
	$(CC) $(FLAGS) $(OBJ) $(DIR) -o qc-dmrg 
	mv qc-dmrg ../bin

$(OBJ):$(HEADERS)

clean:
	rm -f *.o qc-dmrg 

	
