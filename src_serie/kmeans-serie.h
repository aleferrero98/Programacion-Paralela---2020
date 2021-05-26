/**
 * @file 
 * @brief 
 * @author Jeremias Agustinoy y Alejandro Ferrero
 * @version 1.0
 * @date 
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
#define PATH "./inputs/movisA.csv"
#define PATH_2 "./inputs/movisB.csv"
#define TRUE 1
#define FALSE 0
#define CANT_MEANS 4
#define CANT_ITERACIONES 100

u_int64_t CalcLines(char filename[50]);
double * ReadData(char filename[50],u_int64_t size_lines);
double * InitializeMeans(double* items, int cantMeans, double cMin, double cMax);
//double searchMin(const double * items, u_int64_t size_lines);
void searchMinMax(const double * items, u_int64_t size_lines, double* cMin,double* cMax);
//double searchMax(const double * items, u_int64_t size_lines);
double distEuclideana(double x,double y);
double updateMean(double means, double items, int cantItems);
u_int64_t Clasiffy(double* means, double item, int cantMeans);
double * CalculateMeans(int cantMeans, double* items, int cantIterations, int size_lines);


#endif //_KMEANS_SERIE_H_