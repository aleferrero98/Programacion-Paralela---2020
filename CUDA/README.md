# Compilacion y Ejecucion CUDA

##Compilacion
nvcc -g -G  cukmeans.cu -o cukmeans -arch=sm_60

##Ejecucion
./cukmeans

