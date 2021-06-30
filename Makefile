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
KMEANS_CUDA=kmeans-cuda
BIN_CUDA=kmeans_cuda

#Implementaciones
IMPLEMENTACION=naive
#IMPLEMENTACION=efficient

# Opt de compilacion 
CC=gcc
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

#mpi_kmeans:

#cuda_kmeans:

doc: Doxyfile
	mkdir -p $(DOC_DIR)
	doxygen

.PHONY: clean
clean :
	rm  -Rf $(BINARY_DIR) $(DOC_DIR)