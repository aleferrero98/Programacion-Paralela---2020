# K-means MPI

Para paralelizar el algoritmo K-means utilizando MPI se recurrió a una estrategia similar a la empleada en OpenMP, con la diferencia de que ahora se utilizan procesos en lugar de hilos y todas las variables son privadas a cada proceso.
La idea es la siguiente:
- el proceso 0 (p0) es el encargado de realizar la parte serie del algoritmo. Cuenta la cantidad de items y envía dicho valor mediante broadcast a los otros procesos.
- p0 lee los items del archivo, divide la cantidad de items en partes iguales para cada proceso y los reparte con un Scatter (si la division no da entera, los items sobrantes los procesa p0).
- p0 calcula el MAX y MIN de cada feature, inicializa los valores de las medias y se los envía por broadcast a los otros procesos.
- en CalculateMeans, en cada iteración, p0 es el encargado de enviar por broadcast los valores globales de las medias a los otros procesos.
- luego de que cada proceso clasifique sus items, se hace una reducción(suma) de las sumas de los items de cada cluster en cada proceso.
- también se hace una reducción(suma) de la cantidad de items en cada cluster(clusterSizes) que cada proceso clasificó.
- p0 calcula las nuevas medias globales con los valores de las dos reducciones.
- se hace una reducción(AND) para el flag noChange y una reducción(suma) de countChangeItem para determinar si el algoritmo converge o si hay que hacer una iteración más. Además de la reducción se hace un broadcast para que todos los procesos sepan cuándo terminar.
- p0 también es el encargado de ejecutar la función FindClusters en la que se inserta cada item al cluster correspondiente, para lo cual necesita saber la clasificación de todos los items. Esta clasificación la obtiene mediante un Gather del arreglo belongsTo de todos los demás procesos, ya que p0 conoce sólo los items que clasificó.

La idea descrita es la correspondiente al caso más eficiente, en donde cada proceso calcula las medias parciales de todos los clusters y posteriormente se calcula las medias globales.
Para la implementación más simple se debe tener una cantidad de procesos igual 4 (cantidad de medias), ya que cada proceso (del 0 al 3) va a actualizar una media en cada iteración (el proceso 0 con la media 0, proceso 1 con
media 1, etc.). Para este caso, la cantidad de items se distribuye en partes iguales de la misma forma que antes y cada proceso debe clasificar los items que le fueron asignados. Luego de clasificarlos, cada proceso debe actualizar una sola media (la que le corresponde según su rango) y para poder hacerlo, debe recibir los items que poseen los restantes procesos y su clasificación. Entonces, por turnos, cada proceso va enviando sus items y su arreglo de clasificación a los demás. Después de que cada proceso actualice la media que le corresponde, el proceso 0 es el encargado de recibirlas y de enviarlas a todos los demás procesos.

## Compilar y ejecutar

- **Compilar:** mpicc -o <nombre_ejecutable> <archivo.c> -lm
- **Ejecutar:** mpirun -np <NRO_PROCESOS> ejecutable (mpirun -np 4 ./a.out)
