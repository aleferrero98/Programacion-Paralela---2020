# Proyecto Final - Programación Paralela

#### Trabajo Final de la asignatura Programación Paralela - Ingeniería en Computación - FCEFyN - UNC

## Tema: *"Proyecto K-means"*

### Año: 2020

### Autores:
* Ferrero Alejandro Facundo (ale.ferrero@mi.unc.edu.ar)
* Agustinoy Jeremías (jeremias.agustinoy@mi.unc.edu.ar)

## Enunciado 

Dados los archivos de VarianzaA y VarianzaB, aplicar el **algoritmo de k-means** sobre cada uno de estos. Deberá trabajar sobre 4 grupos, los que se corresponden estimativamente con 4 categorias de movimiento. Los valores iniciales pueden asignarlos sobre estimaciones de la media de cada categoria en lugar de aleatoriamente, acorde a una visualizacion de los valores, por ej: 0.5, 4, 9, 15.
Utilice las tres técnicas de paralelismo dadas en clase, **Openmp, MPI y CUDA**.
La finalizacion de la ejecucion debe ser cuando el cambio de componentes de una categoria a otra no supere el 0,1% de los datos o cuando se alcance un umbral de iteraciones. El valor del umbral lo deberá fijar experimentalmente.
Por cada técnica (MPI, CUDA y OpenMP) deberá implementar dos formas de alcanzar la resolución del algoritmo. Puede ser una Naive (inocente) y otra mas eficiente. Por ejemplo, para el caso de MPI, puede dividir un 4 procesos en cada nodo, uno por cada categoria. La opcion de combinar MPI con openMp no cuenta como segunda implementación.
Para Cuda ambas opciones deberán usar memoria shared.


