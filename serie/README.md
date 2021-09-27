# K-means Serie

*K-means* es un algoritmo de clasificación no supervisada (clusterización) que agrupa objetos en *k* grupos basándose en sus características. El agrupamiento se realiza minimizando la suma de distancias entre cada objeto y el centroide de su grupo o cluster.
Para realizar el trabajo práctico se partió implementando en lenguaje C el algoritmo K-means sin paralelizar, el cual fue adapatado de la versión que aparece en la web https://www.geeksforgeeks.org/k-means-clustering-introduction/. Sin embargo, nuestra versión posee dos ligeras diferencias con respecto al de dicha web:
 * Primero se clasifican todos los items y luego se actualizan todas las medias o centroides de cada cluster o grupo (cuando ya se terminó la clasificación de todos los items).
 *  Los valores iniciales para las medias o centroides de cada cluster se inicializan acorde a una visualización de los valores y no aleatoriamente. 

#### Pseudocódigo
> Inicializar k medias con valores equiespaciados dentro del rango de datos.

> Para un número dado de iteraciones:
		Iterar a través de los items:
				Encontrar la media más cercana al item
				Asignar el item a la media
		Actualizar todas las medias con los valores de los items que se encuentran asociadas a cada una de ellas.

Nuestro psudocódigo es el que aparece en sitios como https://www.unioviedo.es/compnum/laboratorios_py/kmeans/kmeans.html o https://es.coursera.org/lecture/mineria-de-datos-introduccion/ejemplo-algoritmo-k-means-d0fgs y permite una mejor paralelización del algoritmo sin modificar los resultados.

Como datos de entrada se generaron aleatoriamente más de 1 millón de items con 3 features (o características) cada uno, para agregarle más complejidad a la parte paralela y así notar los cambios cuando sea paralelizado. Cada linea del archivo csv es un item y sus features están separadas por comas. Cabe aclarar que el código fue desarrollado para que funcione con otras cantidades de items y features también.

## Descripción del algoritmo

 1) Se cuenta la cantidad de items con la función **CalcLines**.
 2) Se leen todos los items del csv con **ReadData**.
 3) Busca el mínimo y máximo valor para cada feature del arreglo items con **searchMinMax**.
 4) Inicializa el arreglo de medias con valores equiespaciados en el rango de datos (**InitializeMeans**).
 5) En la función **CalculateMeans** se recorren todos los items y se los clasifica dentro de un cluster de acuerdo a la distancia euclidiana (con la función **Classify**). Luego se actualizan todas las medias. Esto se realiza para un cierto número de iteraciones o hasta un criterio de convergencia.
 6) Crea una lista de clusters, donde cada cluster es a su vez un arreglo que contiene todos los items que pertenecen a dicho cluster (función **FindClusters**).

El algoritmo finaliza cuando se llega a una cierta cantidad de iteraciones, o cuando entre una iteración y la siguiente la cantidad de items que cambian de cluster es menor al 0.1% del total de los datos.
