/**
 * @file 
 * @brief 
 * @author Jeremias Agustinoy y Alejandro Ferrero
 * @version 1.0
 * @date 26/05/2021
 */

#ifndef _KMEANS_SERIE_H_
#define _KMEANS_SERIE_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <float.h>

#define TAM_STRING 50
//#define PATH "./inputs/movisA.csv"
//#define PATH "./inputs/randomData.csv"
//#define PATH "./inputs/movisB.csv"
#define PATH "./inputs/movisA_2feature.csv"
#define TRUE 1
#define FALSE 0
#define CANT_MEANS 4
#define CANT_ITERACIONES 100
#define TAM_MAX_FILENAME 50     //tamaño max del nombre del archivo en caracteres
#define TAM_LINEA 70
#define CANT_FEATURES 2

u_int64_t CalcLines(char filename[TAM_MAX_FILENAME]);
double** ReadData(char filename[TAM_MAX_FILENAME], u_int64_t size_lines, u_int8_t cant_features);
double** InitializeMeans(u_int16_t cant_means, double* cMin, double* cMax, u_int8_t cant_features);
void searchMinMax(double** items, u_int64_t size_lines, double* minimo, double* maximo, u_int8_t cant_features);
double distanciaEuclidiana(double* x, double* y, int length);
void updateMean(double* mean, double* item, u_int64_t cant_items, u_int8_t cant_features);
u_int64_t Classify(double** means, double* item, int cant_means, int cant_features);
double** CalculateMeans(u_int16_t cant_means, double** items, int cant_iterations, u_int64_t size_lines, u_int64_t* belongsTo, u_int8_t cant_features);
double*** FindClusters(double** items, u_int64_t* belongsTo, u_int64_t cant_items, u_int8_t cant_means, u_int8_t cant_features);
double round(double var);

#endif //_KMEANS_SERIE_H_