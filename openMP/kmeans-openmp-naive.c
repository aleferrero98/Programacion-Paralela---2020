/**
 * @file kmeans-openmp-naive.c
 * @brief  Algoritmo Kmeans de clustering paralelizado con OpenMP.
 * @author Jeremias Agustinoy y Alejandro Ferrero
 * @version 1.0
 * @date 26/05/2021
 */

#include "kmeans-openmp-naive.h"
#include <omp.h>


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

    start2 = omp_get_wtime();
    //se libera la memoria del heap
    #pragma omp parallel num_threads(CANT_MEANS) if(TRUE) shared(size_lines, clusters) default(none)
    {
        #pragma omp for schedule(static) 
        for(int n = 0; n < CANT_MEANS; n++){
            for(u_int64_t m = 0; m < size_lines; m++){
                free(clusters[n][m]);
            } 
            free(clusters[n]);
        }
    }//fin parallel
    free(clusters);
    printf("Duración free(clusters): %f seg\n", omp_get_wtime() - start2);

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
    printf("\033[1;33m >>> Ejecución algoritmo K-means - OpenMP <<<\033[0;37m \n");
    printf("%d threads\n", NUM_THREADS);
    printf("Duración total del programa: %f seg\n", end - start);
    
    return EXIT_SUCCESS;
}


double*** FindClusters(double** items, u_int64_t* belongsTo, u_int64_t cant_items, u_int8_t cant_means, u_int8_t cant_features){

    // clusters es un array de 3 dimensiones, es un conjunto de clusters.
    // cada cluster es un conjunto de items.
    // cada item es un conjunto de features.
    double ***clusters = (double ***) malloc(cant_means * sizeof(double**));
    int indices[cant_means]; //contiene la posicion en la que se debe agregar el proximo item al cluster

    double start;
    start = omp_get_wtime();

    for(u_int8_t n = 0; n < cant_means; n++){
        clusters[n] = (double **) malloc(cant_items * sizeof(double*));
        indices[n] = 0;
           
        for(u_int64_t m = 0; m < cant_items; m++){
            clusters[n][m] = (double *) malloc(cant_features * sizeof(double));
        }
    }

    printf("Duración alloc: %f seg\n", omp_get_wtime() - start);

    start = omp_get_wtime();

    //con critical se obtienen peores resultados
    #pragma omp parallel num_threads(NUM_THREADS) if(TRUE) shared(cant_items, cant_features, belongsTo, clusters, items, indices) default(none)
    {
        //printf("%d\n", omp_get_thread_num());
        #pragma omp for schedule(static) ordered
        for(u_int64_t i = 0; i < cant_items; i++){
            for(u_int8_t j = 0; j < cant_features; j++){ //se cargan todas las features del item al cluster
                //#pragma omp critical(escritura_cluster)
                #pragma omp ordered
                clusters[belongsTo[i]][indices[belongsTo[i]]][j] = items[i][j];
            }
            //#pragma omp critical(escritura_cluster)
            #pragma omp ordered
            indices[belongsTo[i]]++;
        }
    }

    printf("Duración insert clusters: %f seg\n", omp_get_wtime() - start);


    return clusters;
}
/*
double*** FindClusters(double** items, u_int64_t* belongsTo, u_int64_t cant_items, u_int8_t cant_means, u_int8_t cant_features){

    // clusters es un array de 3 dimensiones, es un conjunto de clusters.
    // cada cluster es un conjunto de items.
    // cada item es un conjunto de features.
    double ***clusters = (double ***) malloc(cant_means * sizeof(double**));

    for(u_int8_t n = 0; n < cant_means; n++){
        clusters[n] = (double **) malloc(cant_items * sizeof(double*));
        #pragma omp parallel num_threads(NUM_THREADS) if(cant_items > CANT_MIN_ITEMS) shared(cant_items, cant_features, clusters)
        { 
            #pragma omp for schedule(static)
            for(u_int64_t m = 0; m < cant_items; m++){
                clusters[n][m] = (double *) malloc(cant_features * sizeof(double));
            }
        }
    }

    int *pos = calloc(cant_means, sizeof(int));

    #pragma omp parallel num_threads(NUM_THREADS) if(cant_items > CANT_MIN_ITEMS) shared(cant_means, cant_items, cant_features, belongsTo, clusters, items, pos) default(none)
    {
        double ***clusters_thread = (double ***) malloc(cant_means * sizeof(double**));
        int indices[cant_means]; //contiene la posicion en la que se debe agregar el proximo item al cluster
        for(u_int8_t n = 0; n < cant_means; n++){
            clusters_thread[n] = (double **) malloc(cant_items * sizeof(double*));
            indices[n] = 0;
        }

        #pragma omp for schedule(static) 
        for(u_int64_t i = 0; i < cant_items; i++){
            //se asigna memoria para el item en el cluster que se va a agregar
            clusters_thread[belongsTo[i]][indices[belongsTo[i]]] = (double *) malloc(cant_features * sizeof(double));

            for(u_int8_t j = 0; j < cant_features; j++){ //se cargan todas las features del item al cluster
                clusters_thread[belongsTo[i]][indices[belongsTo[i]]][j] = items[i][j];
            }
            indices[belongsTo[i]]++;
        }

        //cada hilo agrega los items que clasificó al final
        #pragma omp critical
        {
            for(u_int8_t mm = 0; mm < cant_means; mm++){
                clusters[mm][pos[mm]] = clusters_thread[mm][0];
                pos[mm] = indices[mm];
            }
        }
    }//fin parallel
    free(pos);

    return clusters;
}*/

double** CalculateMeans(u_int16_t cant_means, double** items, int cant_iterations, u_int64_t size_lines, u_int64_t* belongsTo, u_int8_t cant_features){
    //Encuentra el minimo y maximo de cada columna (o feature)
    double *cMin, *cMax;
    cMin = (double*) malloc(cant_features * sizeof(double));
    cMax = (double*) malloc(cant_features * sizeof(double));
    searchMinMax(items, size_lines, cMin, cMax, cant_features);

    //define el porcentaje minimo de cambio de items entre clusters para que continue la ejecucion del algoritmo
    double minPorcentaje = 0.001 * (double) size_lines;
    u_int64_t countChangeItem;

    int noChange, j;
    double *item;
    u_int64_t index;
    
    //Inicializa las means (medias) con valores estimativos
    double** means = InitializeMeans(cant_means, cMin, cMax, cant_features);

    //Inicializa los clusters, clusterSizes almacena el numero de items de cada cluster
    u_int64_t *clusterSizes = calloc(cant_means, sizeof(u_int64_t));
    //u_int64_t clusterSizes[cant_means];

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

        int thread_id = 0;

        //se paraleliza solo si la cantidad de items es lo suficientemente grande como para hacerlo
        #pragma omp parallel num_threads(NUM_THREADS)  firstprivate(cant_means, cant_features, size_lines) private(index, item, thread_id) shared(items, belongsTo, noChange, countChangeItem, clusterSizes, sumas_items, means) default(none)
        {       
            thread_id = omp_get_thread_num();
            //printf("Thread %d\n", thread_id); 
            
            //guarda las suma de los valores de los items de cada cluster para despues calcular el promedio (cada thread tiene uno privado)
            double sumas_items_thread[cant_means][cant_features];
            memset(sumas_items_thread, 0, sizeof(double)*cant_means*cant_features);

            //guarda la cantidad de items que se asigno a cada cluster (1 por thread)
            u_int64_t clusterSizes_thread[cant_means]; //privado a cada hilo
            memset(clusterSizes_thread, 0, sizeof(u_int64_t)*cant_means);

        
            #pragma omp for schedule(static) nowait
            for(u_int64_t k = 0; k < size_lines; k++){ //se recorren todos los items
                item = items[k];

                //Clasifica item dentro de un cluster y actualiza las medias correspondientes
                index = Classify(means, item, cant_means, cant_features);

                clusterSizes_thread[index] += 1;
                
                //agrego el valor del item a la suma acumulada del cluster seleccionado                   
                for(int f = 0; f < cant_features; f++){
                    sumas_items_thread[index][f] += item[f];
                }

                //si el Item cambio de cluster
                if(index != belongsTo[k]){
                    #pragma omp atomic write
                        noChange = FALSE;

                    #pragma omp atomic
                        countChangeItem++;

                    belongsTo[k] = index;
                }

            } //NO HAY BARRERA

            //cada hilo calcula la suma de los elementos de cada cluster(sumas_items_thread) 
            //y la cantidad de items en cada cluster(clusterSizes_thread) para calcular despues las medias globales
            for(int m = 0; m < cant_means; m++){
                #pragma omp atomic
                clusterSizes[m] += clusterSizes_thread[m];
                
                for(int f = 0; f < cant_features; f++){
                    #pragma omp atomic
                    sumas_items[m][f] += sumas_items_thread[m][f];
                }
            }
        } //fin parallel

        //calcula las nuevas medias dividiendo las sumas acumuladas por la cantidad de cada cluster
        for(int m = 0; m < cant_means; m++){
            if(clusterSizes[m] == 0) continue; //para evitar divisiones por cero, la media queda en el valor anterior

            for(int f = 0; f < cant_features; f++){
                means[m][f] = sumas_items[m][f] / (double)clusterSizes[m];
            }
        }

        /*printf("Iteracion %d\n", j);
        for(int i = 0; i < cant_means; i++){
            printf("Mean[%d] -> (%lf,%lf,%lf)\n", i, means[i][0], means[i][1], means[i][2]);
        }*/

        //printf("countChangeItem: %lu - minPorcentaje: %lf\n",countChangeItem, minPorcentaje);
        if(noChange || (countChangeItem < minPorcentaje)){
        //if(noChange){
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
