# K-Means CUDA

Para realizar la paralelizacion del algoritmo K-Means con CUDA primero se dividió la funcionalidad del codigo en metodos, destacando los siguientes:
- CalcLines: Calcula cantidad de items.
- ReadData: Lectura de items.
- searchMinMax: Determinacion de Maximos y Minimos.
- InitializeMeans: Inicializar medias.
- CalculateMeans: Calculo iterativo de medias hasta la convergencia.
- FindClusters: Asignación de los items a los diversos clusters.
Una vez que se modularizó el algoritmo, el siguiente objetivo fue determinar cual de ellos duraba mas tiempo en la ejecución serie, de los cuales CalculateMeans fue la función que derivó a mas del 50% del tiempo del programa. 
Por lo que nos enfocamos en paralizar dicha función, ya que la segunda que mas tiempo llevó fue CalcLines junto con ReadData pero como consistian en lectura de datos de un fichero externo, no se paralelizó.
La función **CalculateMeans** se encarga de calcular de forma iterativa las medias hasta que se cumpla la condición de finalización o termine de iterar 100 veces. Como primer paso se determinaron la cantidad de bloques e hilos, partiendo de bloques con poca cantidad de hilos por razones que se explicaran mas adelante.
Se definieron dos funciones de CUDA, una denominada **kMeansClusterAssignment** en la cual hay tanta cantidad de hilos como items trabajando en clasificar el item designado en una media, determinando si cambio respecto a la iteración anterior, en caso de que haya cambiado aumenta la cantidad de items cambiados de forma atomica y finalmente almacena el indice del item dentro de un arreglo en memoria global. 
La segunda función de cuda es denominada **kMeansCentroidUpdate** en la cual se define un arreglo de items por bloque, y otro de los indices designados para los items tambien por bloque, luego se sincronizan los hilos y paso siguiente empieza a trabajar un hilo por bloque, el cual se va a encargar de sumar los items e incrementar la cantidad de items de cada media de todos los hilos del bloque. Finalmente el hilo de cada bloque se encarga de almacenar la suma y la cantidad de items dentro de arreglos globales.
De esta forma se pudo optimizar el tiempo de ejecución del metodo en mas del 50%.

Para la resolución menos eficiente se intentó que en la funcion **kMeansClusterAssignment** el hilo 0 de cada bloque determine si el item cambio basandose en un arreglo de items cambiados anteriores en memoria compartida y el de items cambiados actual, si hubo cambio incrementa una variable que al final la insertará de forma atomica a una variable global, pero el tiempo es mayor que la versión optima y esta limitado a la cantidad de items debido a los arreglos utilizados en CUDA.

## Compilacion y Ejecucion CUDA

### Compilacion
- nvcc -g -G  file.cu -o fileOutput -arch=sm_60

### Ejecucion
- ./fileOutput
