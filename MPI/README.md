PROBLEMAS CON MPI:
- no se puede usar un arreglo de punteros para guardar las matrices ya que en el scatter no sabe cómo copiar los elementos a los otro procesos.
- cuando se envian datos (arreglos, matrices, etc) es necesario que sean zonas contiguas de memoria.
- ver -> https://stackoverflow.com/questions/20031250/mpi-scatter-of-2d-array-and-malloc
- ver -> https://stackoverflow.com/questions/5901476/sending-and-receiving-2d-array-over-mpi

La idea es la siguiente: 
- el proceso 0(p0) calcula cuantos items hay y se lo pasa por broadcast a los otros procesos.
- p0 lee los items del archivo, divide la cantidad en partes iguales y envia cada parte a un proceso distinto con un Scatter (si la division no da entera, los items sobrantes los procesa p0).
- p0 calcula el MAX y MIN de cada feature e inicializa los valores de las MEDIAS y se los envia por broadcasta a los otros procesos.
- en CalculateMeans, p0 siempre hace broadcast de los valores globales actuales de las medias a los otros procesos.
- luego de que cada proceso clsifique sus items, se hace una REDUCCION(suma) de las sumas de los items de cada cluster en cada proceso.
- tambien se hace una reduccion(suma) de la cantidad de items en cada cluster(clusterSizes) que cada proceso clasifico.
- p0 calcula las nuevas medias con los valores de la reduccion.
- se hace una reduccion(AND) para los flag noChange y una reduccion(suma) de countChangeItem para determinar si hay que hacer una iteracion mas o no. Ademas de la reduccion se hace un broadcast para que todos los procesos tengan el dato.
- se envian (BROADCAST) los valores de las medias globales actualizados desde p0 a los otros procesos.

COMANDOS
 - compilar: mpicc kmeans-mpi.c -lm
 - ejecutar:  mpirun -np <NRO_PROCESOS> ejecutable  (mpirun -np 2 ./a.out)  
