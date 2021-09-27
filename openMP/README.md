# K-means OpenMP

Para paralelizar el algoritmo K-means primero se procedió a determinar la función o parte del código que más tiempo demanda. Para eso se utilizó la función de openMP **omp_get_wtime**. Dichas funciones fueron **ReadData** y **CalculateMeans**. La primera, debido a que es lectura de una gran cantidad de items desde un archivo y es de difícil paralelización, no se tuvo en cuenta al paralelizar. En CalculateMeans, la paralelización se realizó en cada iteración, al momento de clasificar todos los items a su media más cercana. Es decir, cada hilo clasifica un parte del total de items. A su vez, cada hilo contribuye a la actualización de la medias de cada cluster, debido a que acumula las sumas de los items de cada cluster que le correspondió clasificar y la cantidad de items en cada cluster. Luego, al final de cada iteración, se juntan los valores acumulados por cada hilo y se calculan las medias globales.

Para el caso de OpenMP se realizaron dos implementaciones, una más sencilla **kmeans-openmp-naive** y otra más optimizada **kmeans-openmp-efficient**.

En particular, para la implementación más sencilla, se paralelizó además dentro de la función FindClusters el bucle que inserta cada item en el cluster correspondiente dentro de la lista de clusters.
En cambio, en la versión más eficiente algunas optimizaciones fueron: se ejecuta en paralelo sólo si hay más de 10000 items, las iteraciones de los bucles se reparten de forma estática entre los hilos y en la función FindClusters sólo se paraleliza la asignación de memoria para la lista de clusters.
