/**
 * @file 
 * @brief 
 * @author Jeremias Agustinoy y Alejandro Ferrero
 * @version 1.0
 * @date 26/05/2021
 */

#ifndef _KMEANS_OPENMP_H_
#define _KMEANS_OPENMP_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <float.h>

//#define PATH "./inputs/movisA.csv"
//#define PATH "./inputs/randomData_3features.csv"
//#define PATH "./inputs/tmp.csv"
#define PATH "./inputs/randomData_5M_3features.csv"

#define TRUE 1
#define FALSE 0
#define CANT_MEANS 4
#define CANT_ITERACIONES 100
#define TAM_MAX_FILENAME 50     //tamaño max del nombre del archivo en caracteres
#define TAM_LINEA 100
#define CANT_MIN_ITEMS 10000  //cantidad minima de items para paralelizar
#define CANT_FEATURES 3

#define NUM_THREADS 4
#define CHUNK 0  //tamaño en el que se divide el trabajo en un bucle for

/**
 * @brief Cuenta la cantidad de lineas del archivo (para definir el tamaño del arreglo items posteriormente)
 * @param filename nombre del archivo
 * @return cantidad de lineas (o items) del archivo
 */
u_int64_t CalcLines(char filename[TAM_MAX_FILENAME]);

/**
 * @brief Lee el archivo indicado y carga el arreglo de items.
 * @param filename string nombre del archivo que contiene los datos
 * @param size_lines cantidad de lineas del archivo
 * @param cant_features cantidad de features de cada item (cantidad de columnas del archivo separadas por comas) 
 * @return arreglo doble con cantidad de filas igual a cantidad de items y cantidad de columnas igual a cantidad de features.
 */
double** ReadData(char filename[TAM_MAX_FILENAME], u_int64_t size_lines, u_int8_t cant_features);

/**
 * @brief Inicializa el arreglo de medias en valores equiespaciados en el rango de datos.
 * @param cant_means cantidad de medias o clusters
 * @param cMin vector con los valores minimos de cada feature
 * @param cMax vector con los valores maximos de cada feature
 * @param cant_features cantidad de features (o columnas) de cada item
 * @return arreglo con las medias (1 por cada cluster).
 * Ejemplo: range: 20 (0 a 19)
 *          cantMeans -> 4
 *          jump: 20 / 4 = 5
 *          means[0] = 0 + 0.5 * 5 = 2.5
 *          means[1] = 0 + 1.5 * 5 = 7.5
 *          means[2] = 0 + 2.5 * 5 = 12.5
 *          means[3] = 0 + 3.5 * 5 = 17.5
 */
double** InitializeMeans(u_int16_t cant_means, double* cMin, double* cMax, u_int8_t cant_features);

/**
 * @brief Busca el minimo y maximo valor para cada feature del arreglo items.
 * @param items datos a clasificar
 * @param size_lines cantidad de items
 * @param minimo arreglo de los valores minimos de cada feature
 * @param maximo arreglo de los valores maximos de cada feature
 * @param cant_features cantidad de caracteristicas que tiene cada item
 */
void searchMinMax(double** items, u_int64_t size_lines, double* minimo, double* maximo, u_int8_t cant_features);

/**
 * @brief Calcula distancia euclidiana entre dos vectores.
 * @param x primer vector (item)
 * @param y segundo vector (item)
 * @param length longitud del vector
 * @return distancia euclidiana entre ambos vectores.
 */
double distanciaEuclidiana(double* x, double* y, int length);

/**
 * @brief Actualiza el valor de la media incorporando un nuevo item al cluster. 
 * @param mean es la media (arreglo) que se va a actualizar.
 * @param item es el nuevo valor que se quiere introducir en el cluster de esa media.
 * @param cant_items es la cantidad de item en el cluster de esa media.
 * @param cant_features cantidad de caracteristicas de un item.
 * Formula que se aplica para calcular la nueva media: m = (m*(n-1)+x)/n
 */
void updateMean(double* mean, double* item, u_int64_t cant_items, u_int8_t cant_features);

/**
 * @brief Clasifica un item dentro de una media (o cluster), de acuerdo a la distancia euclidiana.
 * @param means arreglo de medias
 * @param item item a clasificar
 * @param cant_means cantidad de medias o clusters 
 * @param cant_features cantidad de caracteristicas de cada item
 * @return el indice de la media a la que se asocio el item.
 */
u_int64_t Classify(double** means, double* item, int cant_means, int cant_features);

/**
 * @brief Calcula los valores de la media de cada cluster. Itera por todos los items,
 * los clasifica al cluster mas cercano y actualiza la media del cluster.
 * El algoritmo se repite para un numero fijo de iteraciones y si entre dos iteraciones
 * no hay cambios en la clasificacion de ningun item, se para el proceso (el algoritmo
 * encontro la solucion optima).
 * Es la funcion principal del algoritmo.
 * @param cant_means cantidad de medias o clusters.
 * @param items arreglo de items a clasificar.
 * @param cant_iterations cantidad maxima de iteraciones del algoritmo
 * @param size_lines cantidad de items
 * @param belongsTo array que contiene el numero de cluster al que pertenece cada item
 * @param cant_features cantidad de caracteristicas de los items.
 * @return arreglo de medias de todos los clusters.
 */
double** CalculateMeans(u_int16_t cant_means, double** items, int cant_iterations, u_int64_t size_lines, u_int64_t* belongsTo, u_int8_t cant_features);

/**
 * @brief Crea una lista de clusters, donde cada cluster es a su vez un arreglo que contiene 
 * todos los items que pertenecen a dicho cluster.
 * @param items items a clasificar
 * @param belongsTo arreglo que contiene el indice del cluster al que pertenece cada item
 * @param cant_intems cantidad de items
 * @param cant_means cantidad de clusters o grupos
 * @param cant_features cantidad de caracteristicas de cada item.
 * @return arreglo de clusters, c/u con sus items
 */
double*** FindClusters(double** items, u_int64_t* belongsTo, u_int64_t cant_items, u_int8_t cant_means, u_int8_t cant_features);

/**
 * @brief redondea un numero de punto flotante a 3 cifras despues de la coma.
 * @param var numero a redondear
 * @return numero con 3 cifras decimales
 * Ejemplo: 37.66666 * 1000 = 37666.66
 *          37666.66 + .5 = 37667.16    for rounding off value
 *          then type cast to int so value is 37667
 *          then divided by 1000 so the value converted into 37.67
 */
double round(double var);

#endif //_KMEANS_OPENMP_H_