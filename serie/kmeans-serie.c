/**
 * @file kmeans-serie.c
 * @brief Algoritmo Kmeans de clustering, se utiliza para agrupar items con caracteristicas similares.
 * @author Jeremias Agustinoy y Alejandro Ferrero
 * @version 1.0
 * @date 26/05/2021
 */

#include "kmeans-serie.h"
#include <omp.h> //para la funcion omp_get_wtime


int main(void) {
    double start, end, start2; 

    start = omp_get_wtime(); 
    double ***clusters;
    u_int64_t *belongsTo;  //Define un arreglo para almacenar el indice del cluster al que pertenece cada item
    
    start2 = omp_get_wtime(); 
    u_int64_t size_lines = CalcLines(PATH);
    double **items = ReadData(PATH, size_lines, CANT_FEATURES);
    printf("Duración de CalcLines + ReadData: %f seg\n\n", omp_get_wtime() - start2);

    //Define un arreglo para almacenar el indice del cluster al que pertenece cada item
    belongsTo = calloc(size_lines, sizeof(u_int64_t));
    
    start2 = omp_get_wtime(); 
    double **means = CalculateMeans(CANT_MEANS, items, CANT_ITERACIONES, size_lines, belongsTo, CANT_FEATURES);
    printf("Duración de CalculateMeans: %f seg\n", omp_get_wtime() - start2);

    start2 = omp_get_wtime();
    clusters = FindClusters(items, belongsTo, size_lines, CANT_MEANS, CANT_FEATURES);
    printf("Duración de FindClusters: %f seg\n", omp_get_wtime() - start2);
    
    printf("\nValores de las medias finales:\n");
    for(int i = 0; i < CANT_MEANS; i++){
        printf("Mean[%d] -> (%lf,%lf,%lf)\n", i, means[i][0], means[i][1], means[i][2]);
    }

    /*printf("clusters: (%lf,%lf,%lf) - (%lf,%lf,%lf) - (%lf,%lf,%lf) - (%lf,%lf,%lf)\n", 
                            clusters[0][0][0], clusters[0][0][1], clusters[0][0][2],
                                clusters[1][0][0], clusters[1][0][1], clusters[1][0][2],
                                 clusters[2][0][0],  clusters[2][0][1],  clusters[2][0][2],
                                  clusters[3][0][0],  clusters[3][0][1],  clusters[3][0][2]);*/

    //se libera la memoria del heap
    for(int n = 0; n < CANT_MEANS; n++){
        for(u_int64_t m = 0; m < size_lines; m++){
            free(clusters[n][m]);
        } 
        free(clusters[n]);
    }
    free(clusters);

    free(belongsTo);
    for(int n = 0; n < CANT_MEANS; n++){
        free(means[n]);
    }
    free(means);
    for(int n = 0; n < CANT_FEATURES; n++){
        free(items[n]);
    }
    free(items);

    end = omp_get_wtime(); 
    printf("\033[1;33m >>> Ejecución algoritmo K-means Serie <<<\033[0;37m \n");
    printf("Duración total del programa: %f seg\n", end - start);
    
    return EXIT_SUCCESS;
}

double*** FindClusters(double** items, u_int64_t* belongsTo, u_int64_t cant_items, u_int8_t cant_means, u_int8_t cant_features){

    // clusters es un array de 3 dimensiones, es un conjunto de clusters.
    // cada cluster es un conjunto de items.
    // cada item es un conjunto de features.
    double ***clusters = (double ***) malloc(cant_means * sizeof(double**));
    int indices[cant_means]; //contiene la posicion en la que se debe agregar el proximo item al cluster

    for(u_int8_t n = 0; n < cant_means; n++){
        clusters[n] = (double **) malloc(cant_items * sizeof(double*));
        indices[n] = 0;
        for(u_int64_t m = 0; m < cant_items; m++){
            clusters[n][m] = (double *) malloc(cant_features * sizeof(double));
        }
    }

    for(u_int64_t i = 0; i < cant_items; i++){
        for(u_int8_t j = 0; j < cant_features; j++){ //se cargan todas las features del item al cluster
            clusters[belongsTo[i]][indices[belongsTo[i]]][j] = items[i][j];
        }
        indices[belongsTo[i]]++;
    }

    return clusters;
}

double** CalculateMeans(u_int16_t cant_means, double** items, int cant_iterations, u_int64_t size_lines, u_int64_t* belongsTo, u_int8_t cant_features){
    //Encuentra el minimo y maximo de cada columna (o feature)
    double *cMin, *cMax;
    double start;

    cMin = (double*) malloc(cant_features * sizeof(double));
    cMax = (double*) malloc(cant_features * sizeof(double));
    start = omp_get_wtime();
    searchMinMax(items, size_lines, cMin, cMax, cant_features);
    printf("Duración de searchMinMax: %f seg\n", omp_get_wtime() - start);

    //define el porcentaje minimo de cambio de items entre clusters para que continue la ejecucion del algoritmo
    double minPorcentaje = 0.001 * (double) size_lines;
    u_int64_t countChangeItem;

    int noChange, j;
    double *item;
    u_int64_t index;

    start = omp_get_wtime();
    //Inicializa las means (medias) con valores estimativos
    double** means = InitializeMeans(cant_means, cMin, cMax, cant_features);
    printf("Duración de InitializeMeans: %f seg\n\n", omp_get_wtime() - start);

    //Inicializa los clusters, clusterSizes almacena el numero de items de cada cluster
    u_int64_t* clusterSizes = calloc(cant_means, sizeof(u_int64_t));

    //guarda las suma de los valores de los items de cada cluster para despues calcular el promedio
    double sumas_items[cant_means][cant_features]; 

    //Calcula las medias
    for(j = 0; j < cant_iterations; j++) {
        
        //Si no ocurrio un cambio en el cluster, se detiene
        noChange = TRUE;
        countChangeItem = 0;

        //Resetea el clusterSizes a 0 para cada una de las medias
        memset(clusterSizes, 0, sizeof(u_int64_t)*cant_means);
        memset(sumas_items, 0, sizeof(double)*cant_means*cant_features);

        for(u_int64_t k = 0; k < size_lines; k++) { //se recorren todos los items
            item = items[k];

            //Clasifica item dentro de un cluster y actualiza las medias correspondientes
            index = Classify(means, item, cant_means, cant_features);

            clusterSizes[index] += 1;
            //cSize = clusterSizes[index]; //cant de items del cluster
            //updateMean(means[index], item, cSize, cant_features);

            //agrego el valor del item a la suma acumulada del cluster seleccionado
            for(int f = 0; f < cant_features; f++){
                sumas_items[index][f] += item[f];
            }

            //si el Item cambio de cluster
            if(index != belongsTo[k]){
                noChange = FALSE;
                countChangeItem++;
                belongsTo[k] = index;
            }

        }

        //calcula las nuevas medias dividiendo las sumas acumuladas por la cantidad de cada cluster
        for(int m = 0; m < cant_means; m++){
            if(clusterSizes[m] == 0) continue; //para evitar divisiones por cero, la media queda en el valor anterior

            for(int f = 0; f < cant_features; f++){
                means[m][f] = sumas_items[m][f] / (double)clusterSizes[m];
            }
        }

        printf("Iteracion %d\n", j);
        /*
        for(int i = 0; i < cant_means; i++){
            printf("Mean[%d] -> (%lf,%lf,%lf)\n", i, means[i][0], means[i][1], means[i][2]);
        }*/
        /*
        for (int m = 0; m < cantMeans; ++m) {
            printf("Cluster[%d]: %lu\n", m, clusterSizes[m]);
        }*/

        //printf("countChangeItem: %lu - minPorcentaje: %lf\n",countChangeItem, minPorcentaje);
        //if(noChange){ 
        if(noChange || (countChangeItem < minPorcentaje)){
            break;
        }
    }

    printf("\n>>> Cantidad de items en cada cluster <<<\n");
    for (int m = 0; m < cant_means; m++) {
        printf("Cluster[%d]: %lu\n", m, clusterSizes[m]);
    }
    printf("Cantidad de iteraciones: %d\n", j);

    free(clusterSizes);
    free(cMin);
    free(cMax);

    return means;
}


u_int64_t Classify(double** means, double* item, int cant_means, int cant_features){
    double minimun = DBL_MAX;
    int index = -1;
    double distance;

    for(int i = 0; i < cant_means; i++){
        //calcula la distancia de un item a la media
        distance = distanciaEuclidiana(item, means[i], cant_features);
        if(distance < minimun){
            minimun = distance;
            index = i;
        }
    }
    return (u_int64_t) index;
}


double distanciaEuclidiana(double* x, double* y, int length){
    double distancia = 0;
    for(int i = 0; i < length; i++){
        distancia += pow((x[i] - y[i]), 2);
    }
   
    return sqrt(distancia);
}


void updateMean(double* mean, double* item, u_int64_t cant_items, u_int8_t cant_features){
    double m;

    for(u_int8_t i = 0; i < cant_features; i++){
        m = mean[i];
        m = (m * ((double) cant_items - 1) + item[i]) / (double) cant_items;
        mean[i] = round(m); //se redondea a 3 cifras decimales
    } 
}


double round(double var){
    double value = (int)(var * 1000 + .5);
    return (double)value / 1000;
}


double** ReadData(char filename[TAM_MAX_FILENAME], u_int64_t size_lines, u_int8_t cant_features){
    
    FILE *file = fopen(filename, "r");
    rewind(file);

    //Definimos un arreglo de arreglos (cada item consta de 2 features)
    double** items = (double **) malloc(size_lines * sizeof(double*));
    
    for(u_int64_t n = 0; n < size_lines; n++){
        items[n] = (double *) malloc(cant_features * sizeof(double));
    }

    char* line = calloc(TAM_LINEA, sizeof(char));
    double feature;
    u_int64_t i = 0, j = 0;
    char* ptr;

    while(fgets(line, TAM_LINEA, file)){
        j = 0;
        char *item = strstr(line, ","); //se ignora el primer elemento del archivo (indice)
        item++;
        if(item != NULL && strcmp(item, "values\n") && strcmp(item, "\n")){ //Para recortar la cadena y tomar solo el segundo dato
           // item[strlen(item)-1] = '\0';
            char *token = strtok(item, ","); //separa los elementos de la linea por comas
            while(token != NULL){
                feature = strtod(token, &ptr); //Pasaje a double
                items[i][j] = feature; //Almacenamiento en item
                j++;
                token = strtok(NULL, ","); //busco el siguiente token
            }
            i++;
        }
    }
    free(line);
    fclose(file);

    return items;
}


double** InitializeMeans(u_int16_t cant_means, double* cMin, double* cMax, u_int8_t cant_features){
    
    double **means = (double **) malloc(cant_means * sizeof(double*));
    for(u_int16_t n = 0; n < cant_means; n++){
        means[n] = (double *) malloc(cant_features * sizeof(double));
    }

    //definimos el salto de un valor de media al siguiente
    double *jump = (double *) malloc(cant_features * sizeof(double));
    for(u_int8_t n = 0; n < cant_features; n++){
        jump[n] = (double) (cMax[n] - cMin[n]) / cant_means;
    }

   printf("\nValores de las medias iniciales:\n");
    for(u_int16_t i = 0; i < cant_means; i++){
        for(u_int8_t j = 0; j < cant_features; j++){
            means[i][j] = cMin[j] + (0.5 + i) * jump[j];
        }
        printf("Mean[%d] -> (%lf,%lf,%lf)\n", i, means[i][0], means[i][1],  means[i][2]);
    }

    free(jump);
    return means;
}

// PODRIAMOS HACER QUE CALCULE LA CANTIDAD DE FEATURES TAMBIEN

u_int64_t CalcLines(char filename[TAM_MAX_FILENAME]) {
    FILE *f = fopen(filename, "r");
    u_int64_t cant_lines = 0; 
    char* cadena = calloc(TAM_LINEA, sizeof(char));
    char* valor;
    while(fgets(cadena, TAM_LINEA, f)){
        valor = strstr(cadena, ",");
        valor++;
        if(valor != NULL && strcmp(valor,"values\n") && strcmp(valor,"\n")){
            cant_lines++;
        }
    }
    free (cadena);
    fclose(f);
    printf("Cantidad de items: %ld\n", cant_lines);

    return cant_lines;
}


void searchMinMax(double** items, u_int64_t size_lines, double* minimo, double* maximo, u_int8_t cant_features){

    //Define el maximo como el minimo valor de tipo DOUBLE y el minimo como el maximo valor de tipo DOUBLE
    for(int n = 0; n < cant_features; n++){
        maximo[n] = DBL_MIN;
        minimo[n] = DBL_MAX;
    }
    
    for(u_int64_t i = 0; i < size_lines; i++){  //recorremos cada item
        for(u_int8_t j = 0; j < cant_features; j++){  //recorremos cada feature
            if(items[i][j] < minimo[j]){
                minimo[j] = items[i][j];
            }
            if(items[i][j] > maximo[j]){
                maximo[j] = items[i][j];
            }
        }
    }

    printf("maximos: %lf, %lf, %lf\n", maximo[0], maximo[1], maximo[2]);
    printf("minimos: %lf, %lf, %lf\n", minimo[0], minimo[1], minimo[2]);
}
