/**
 * @file 
 * @brief 
 * @author Jeremias Agustinoy y Alejandro Ferrero
 * @version 1.0
 * @date 26/05/2021
 */

#include "kmeans-serie.h"
#include <omp.h> //para la funcion omp_get_wtime

int main() {
    double start, end; 

    start = omp_get_wtime(); 
   // double cMin,cMax;
    double **clusters;
    u_int64_t* belongsTo;  //Define un arreglo para almacenar el indice del cluster al que pertenece cada item
    u_int64_t size_lines = CalcLines(PATH);
    double* items = ReadData(PATH, size_lines);

    //Define un arreglo para almacenar el indice del cluster al que pertenece cada item
    belongsTo = calloc(size_lines, sizeof(u_int64_t));
    
    double* means = CalculateMeans(CANT_MEANS, items, CANT_ITERACIONES, size_lines, belongsTo);
    clusters = FindClusters(items, belongsTo, size_lines);
    
    for(int n = 0; n < CANT_MEANS; n++){
        free(clusters[n]);
    }
    free(clusters);
    free(belongsTo);
    free(means);
    free(items);

    end = omp_get_wtime(); 
    printf("\033[1;33m >>> Ejecución algoritmo K-means Serie <<<\033[0;37m \n");
    printf("Duración total del programa: %f seg\n", end - start);
    
    return EXIT_SUCCESS;
}

double** FindClusters(double *items, u_int64_t *belongsTo, u_int64_t cantItems){
   // double clusters[CANT_MEANS][cantItems];
    double **clusters = (double **) malloc(CANT_MEANS * sizeof(double*));
    int indices[CANT_MEANS];
    
    for(int n = 0; n < CANT_MEANS; n++){
        clusters[n] = (double *) malloc(cantItems * sizeof(double));
        indices[n] = 0;
    }

    for(u_int64_t i = 0; i < cantItems; i++){
        clusters[belongsTo[i]][indices[belongsTo[i]]] = items[i];
        indices[belongsTo[i]]++;
        //printf("belong: %lu\n", belongsTo[i]);
    }

    return clusters;
}

double* CalculateMeans(u_int16_t cantMeans, double* items, int cantIterations, u_int64_t size_lines, u_int64_t* belongsTo){
    //Encuentra el minimo y maximo de la columna
    double cMin,cMax;
    searchMinMax(items, size_lines, &cMin, &cMax);
    //double cMin = searchMin(items,size_lines);
    //double cMax = searchMax(items,size_lines);

    double minPorcentaje = 0.001 * (double) size_lines;
    u_int64_t countChangeItem;

    //Definicion de variables
    int noChange, j;
    double item;
    u_int64_t cSize, index;

    //Inicializa las means (medias) con valores estimativos
    double* means = InitializeMeans(cantMeans, cMin, cMax);

    //Inicializa los clusters, el arreglo almacena el numero de items
    u_int64_t* clusterSizes = calloc(cantMeans, sizeof(u_int64_t));

    //Define un arreglo para almacenar el indice del cluster al que pertenece cada item
    //belongsTo = calloc(size_lines, sizeof(u_int64_t));

    //Calcula las medias
    for (j = 0; j < cantIterations; j++) {
        
        //Si no ocurrio un cambio en el cluster, se detiene
        noChange = TRUE;
        countChangeItem = 0;

        //Resetea el clusterSizes a 0 para cada una de las medias
        memset(clusterSizes, 0, sizeof(u_int64_t)*CANT_MEANS);

        for (u_int64_t k = 0; k < size_lines; ++k) {
            item = items[k];

            //Clasifica item dentro de un cluster y actualiza las medias correspondientes
            index = Clasiffy(means, item, cantMeans);

            clusterSizes[index] += 1;
            cSize = clusterSizes[index];
            //printf("Later - Mean[%lu] = %f\n", index,means[index]);
            //printf("Result:%f\n",updateMean(means[index], item, cSize));
            means[index] = updateMean(means[index], item, cSize);
            //printf("After - Mean[%lu] = %f\n", index,means[index]);

            //Item cambio el cluster
            if(index != belongsTo[k]){
                noChange = FALSE;
                countChangeItem++;
            }

            belongsTo[k] = index;

        }
        //printf("Iteracion %d\n", j);
        /*for (int n = 0; n < cantMeans; ++n) {
            printf("Means[%d]: %f\n", n, means[n]);
        }

        for (int m = 0; m < cantMeans; ++m) {
            printf("Cluster[%d]: %lu\n", m, clusterSizes[m]);
        }*/

        //printf("countChangeItem: %lu - minPorcentaje: %lf\n",countChangeItem, minPorcentaje);
        /*if(noChange || (countChangeItem < minPorcentaje)){
            break;
        }*/
        if(noChange){
            break;
        }
    }

    printf(">>> Cantidad de items en cada cluster <<<\n");
    for (int m = 0; m < cantMeans; m++) {
        printf("Cluster[%d]: %lu\n", m, clusterSizes[m]);
    }
    printf("Cantidad de iteraciones: %d\n", j);

    free(clusterSizes);
   // free(belongsTo);
    return means;
}

/* Clasifica un item dentro de una media, de acuerdo a la distancia euclideana */
u_int64_t Clasiffy(double* means, double item, int cantMeans){
    double minimun = DBL_MAX;
    int index = -1;
    double distance;

    for(int i = 0; i<cantMeans; i++){
        distance = distEuclideana(item,means[i]);
        if(distance < minimun){
            minimun = distance;
            index = i;
        }
    }
    return (u_int64_t) index;
}

/* Calcula distancia euclideana entre dos valores */
double distEuclideana(double x, double y){
    return sqrt(pow((x-y),2));
}

/*
Actualiza el valor de la media. 
- mean es la media que se va a cambiar,
- item es el nuevo valor que se quiere introducir en el cluster de esa media,
- cantItems es la cantidad

Formula: m = (m*(n-1)+x)/n
*/
double updateMean(double mean, double item, u_int64_t cantItems){
    double m;
    m=mean;
    m=(m*((double) cantItems-1)+item)/(double) cantItems;
    //mean = round(m);
    return m;
}

/* Lee el archivo y arma el arreglo de Items */
double* ReadData(char filename[50], u_int64_t size_lines){
    
    FILE *f = fopen(filename,"r");
    //u_int64_t size_lines = CalcLines(f);
    rewind(f);

    //Definimos arreglo
    double* items = malloc(size_lines * sizeof(double));

    char* line = calloc(TAM_STRING,sizeof(char));
    double feature;
    u_int64_t i=0;
    char* ptr;

    while(fgets(line,TAM_STRING,f)){
        char *item = strstr(line,",");
        item++;
        if(item != NULL && strcmp(item,"values\n") && strcmp(item,"\n")){ //Para recortar la cadena y tomar solo el segundo dato
           // item[strlen(item)-1] = '\0';
            feature = strtod(item,&ptr); //Pasaje a double
            items[i] = feature; //Almacenamiento en item
           // if(i == 0 || i == 1 || i == 2 || i == 3) printf("item[%lu]: %.16f\n",i,items[i]);
            i++;
        }
    }

    free(line);
    fclose(f);

    return items;
}

/* Inicializa el arreglo de medias en valores equiespaciados en el rango de datos */
double* InitializeMeans(u_int16_t cantMeans, double cMin, double cMax){
    double* means = malloc(cantMeans * sizeof(double));
    double range = cMax - cMin;
    double jump = range / cantMeans;

    /*
    Ejemplo: range: 20
             jump: 20 / 4 = 5
             cantMeans -> PAR (4)
                means[0] = 0 + 0.5 * 5 = 2.5
                means[1] = 0 + 1.5 * 5 = 7.5
                means[2] = 0 + 2.5 * 5 = 12.5
                means[3] = 0 + 3.5 * 5 = 17.5
             
             range:20
             jump: 20 / 3 = 6.67
             cantMeans -> IMPAR(3)
                means[0] = 0 + 0.5 * 6.67 = 3.34
                means[1] = 0 + 1.5 * 6.67 = 10.01
                means[2] = 0 + 2.5 * 6.67 = 16.67

    */
    for(int i = 0; i < cantMeans; i++){
        means[i] = cMin + (0.5 + i) * jump;
        printf("Mean[%d]: %lf\n", i, means[i]);
    }
    return means;
}

/* Calcula la cantidad de lineas del archivo */
u_int64_t CalcLines(char filename[50]) {
    FILE *f = fopen(filename,"r");
    u_int64_t cant_lines = 0; 
    char* cadena = calloc(TAM_STRING, sizeof(char));
    char* valor;
    while(fgets(cadena, TAM_STRING, f)){
        valor = strstr(cadena,",");
        valor++;
        //printf("valor: %s\n",valor);
        if(valor != NULL && strcmp(valor,"values\n") && strcmp(valor,"\n")){
            //printf("line:%s\n",cadena);
            cant_lines ++;
        }
    }
    free (cadena);
    fclose(f);
    //printf("cant_lines %ld\n",cant_lines);
    return cant_lines;
}

/*Encontramos el item minimo del arreglo ITEMS
double searchMin(const double * items, u_int64_t size_lines){
    //Define el minimo como el maximo valor de tipo DOUBLE
    double minimal = DBL_MAX;

    for(u_int64_t i = 0; i < size_lines; i++){
        if(items[i] < minimal){
            minimal = items[i];
        }
    }
    return minimal;
}*/

/*Encontramos el item maximo del arreglo ITEMS
double searchMax(const double * items, u_int64_t size_lines){
    //Define el maximo como el minimo valor de tipo LONG
    double maximal = DBL_MIN;
    for(u_int64_t i = 0; i < size_lines; i++){
        if(items[i] > maximal){
            maximal = items[i];
        }
    }
    return maximal;
}*/

/*Encontramos minimo y maximo del arreglo ITEMS*/
void searchMinMax(const double * items, u_int64_t size_lines, double* cMin,double* cMax){
    //Define el maximo como el minimo valor de tipo DOUBLE
    double maximal = DBL_MIN;
    //Define el minimo como el maximo valor de tipo DOUBLE
    double minimal = DBL_MAX;
    
    for(u_int64_t i = 0; i < size_lines; i++){
        if(items[i] > maximal){
            maximal = items[i];
        }
        if(items[i] < minimal){
            minimal = items[i];
        }
    }
    *cMin = minimal;
    *cMax = maximal;
}