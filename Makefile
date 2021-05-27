# Directorios
SOURCE_SERIE=src_serie
SOURCE_OPENMP=src_openmp
SOURCE_MPI=src_mpi
SOURCE_CUDA=src_cuda
BINARY_DIR=bin
DOC_DIR=documentacion

#Binarios
KMEANS_SERIE=kmeans-serie
BIN_SERIE=kmeans_serie
KMEANS_OPENMP=kmeans-openmp
BIN_OPENMP=kmeans_openmp
KMEANS_MPI=kmeans-mpi
BIN_MPI=kmeans_mpi
KMEANS_CUDA=kmeans-cuda
BIN_CUDA=kmeans_cuda

#Implementaciones
IMPLEMENTACION=naive
#IMPLEMENTACION=efficient

# Opt de compilacion 
CC=gcc
PROFILER= scorep --openmp #--verbose
CFLAGS= -g -Wall -pedantic -Wextra -Wconversion -march=native -O0 #-Werror

all: kmeans_serie kmeans_openmp #kmeans_mpi kmeans_cuda doc

kmeans_serie: $(SOURCE_SERIE)/$(KMEANS_SERIE).c  $(SOURCE_SERIE)/$(KMEANS_SERIE).h
	mkdir -p $(BINARY_DIR)
	$(CC) $(CFLAGS) -o $(BINARY_DIR)/$(BIN_SERIE) $(SOURCE_SERIE)/$(KMEANS_SERIE).c -lm

kmeans_openmp: $(SOURCE_OPENMP)/$(KMEANS_OPENMP)-$(IMPLEMENTACION).c  $(SOURCE_OPENMP)/$(KMEANS_OPENMP)-$(IMPLEMENTACION).h
	mkdir -p $(BINARY_DIR)
	$(CC) $(CFLAGS) -o $(BINARY_DIR)/$(BIN_OPENMP)_$(IMPLEMENTACION) $(SOURCE_OPENMP)/$(KMEANS_OPENMP)-$(IMPLEMENTACION).c -lm -fopenmp
	$(PROFILER) $(CC) $(CFLAGS) -o $(BINARY_DIR)/$(BIN_OPENMP)_$(IMPLEMENTACION) $(SOURCE_OPENMP)/$(KMEANS_OPENMP)-$(IMPLEMENTACION).c -lm -fopenmp

#kmeans_mpi:

#kmeans_cuda:

doc: Doxyfile
	mkdir -p $(DOC_DIR)
	doxygen

.PHONY: clean
clean :
	rm  -Rf $(BINARY_DIR) $(DOC_DIR)