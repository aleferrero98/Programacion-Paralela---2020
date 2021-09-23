# Directorios
BINARY_DIR=bin
DOC_DIR=documentacion
SERIE_DIR=serie
OPENMP_DIR=openMP
MPI_DIR=MPI
CUDA_DIR=CUDA

#Binarios
KMEANS_SERIE=kmeans-serie
BIN_SERIE=kmeans_serie
KMEANS_OPENMP=kmeans-openmp
BIN_OPENMP=kmeans_openmp
KMEANS_MPI=kmeans-mpi
BIN_MPI=kmeans_mpi
KMEANS_CUDA=cukmeans
BIN_CUDA=kmeans_cuda

#Implementaciones
IMPLEMENTACION=naive
#IMPLEMENTACION=efficient

# Opt de compilacion 
CC=gcc
MPICC=mpicc
NUM_PROCS=4
NVCC=nvcc
PROFILER= scorep --openmp #--verbose
CFLAGS= -g -Wall -pedantic -Wextra -Wconversion -march=native -O0 #-Werror

all: serie_kmeans openmp_kmeans #mpi_kmeans cuda_kmeans doc

serie_kmeans: $(SERIE_DIR)/$(KMEANS_SERIE).c  $(SERIE_DIR)/$(KMEANS_SERIE).h
	mkdir -p $(BINARY_DIR)/$(SERIE_DIR)
	$(CC) $(CFLAGS) -o $(BINARY_DIR)/$(SERIE_DIR)/$(BIN_SERIE) $(SERIE_DIR)/$(KMEANS_SERIE).c -lm -fopenmp

openmp_kmeans: $(OPENMP_DIR)/$(KMEANS_OPENMP)-$(IMPLEMENTACION).c  $(OPENMP_DIR)/$(KMEANS_OPENMP)-$(IMPLEMENTACION).h
	mkdir -p $(BINARY_DIR)/$(OPENMP_DIR)
	$(CC) $(CFLAGS) -o $(BINARY_DIR)/$(OPENMP_DIR)/$(BIN_OPENMP)_$(IMPLEMENTACION)  $(OPENMP_DIR)/$(KMEANS_OPENMP)-$(IMPLEMENTACION).c -lm -fopenmp
# $(PROFILER) $(CC) $(CFLAGS) -o $(BINARY_DIR)/$(OPENMP_DIR)/$(BIN_OPENMP)_$(IMPLEMENTACION)  $(OPENMP_DIR)/$(KMEANS_OPENMP)-$(IMPLEMENTACION).c -lm -fopenmp

mpi_kmeans2: $(MPI_DIR)/$(KMEANS_MPI)-$(IMPLEMENTACION).c  $(MPI_DIR)/$(KMEANS_MPI)-$(IMPLEMENTACION).h
	mkdir -p $(BINARY_DIR)/$(MPI_DIR)
	$(MPICC) $(CFLAGS) -o $(BINARY_DIR)/$(MPI_DIR)/$(BIN_MPI)_$(IMPLEMENTACION)  $(MPI_DIR)/$(KMEANS_MPI)-$(IMPLEMENTACION).c -lm 

mpi_kmeans: $(MPI_DIR)/$(KMEANS_MPI).c  $(MPI_DIR)/$(KMEANS_MPI).h
	mkdir -p $(BINARY_DIR)/$(MPI_DIR)
	$(MPICC) $(CFLAGS) -o $(BINARY_DIR)/$(MPI_DIR)/$(BIN_MPI)  $(MPI_DIR)/$(KMEANS_MPI).c -lm 

mpi_run:
	mpirun -np $(NUM_PROCS) $(BINARY_DIR)/$(MPI_DIR)/$(BIN_MPI)

cuda_kmeans: $(CUDA_DIR)/$(KMEANS_CUDA).cu
	mkdir -p $(BINARY_DIR)/$(CUDA_DIR)
	$(NVCC) -g -G  $(CUDA_DIR)/$(KMEANS_CUDA).cu -o $(BINARY_DIR)/$(CUDA_DIR)/$(BIN_CUDA) -arch=sm_60

doc: Doxyfile
	mkdir -p $(DOC_DIR)
	doxygen

.PHONY: clean
clean :
	rm  -Rf $(BINARY_DIR) $(DOC_DIR)
