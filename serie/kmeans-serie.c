/**
 * @file kmeans-serie.c
 * @brief Algoritmo Kmeans de clustering, se utiliza para agrupar items con caracteristicas similares.
 * @author Jeremias Agustinoy y Alejandro Ferrero
 * @version 1.0
 * @date 26/05/2021
 */

#include "kmeans-serie.h"
#include <omp.h> //para la funcion omp_get_wtime

/**
 * @brief
 * @return codigo de retorno al SO
 */
int main(void) {
    double start, end, start2; 

    start = omp_get_wtime(); 
   // double cMin,cMax;
    double ***clusters;
    u_int64_t *belongsTo;  //Define un arreglo para almacenar el indice del cluster al que pertenece cada item
    u_int64_t size_lines = CalcLines(PATH);
    double **items = ReadData(PATH, size_lines, CANT_FEATURES);

    //Define un arreglo para almacenar el indice del cluster al que pertenece cada item
    belongsTo = calloc(size_lines, sizeof(u_int64_t));
    
    start2 = omp_get_wtime(); 
    double **means = CalculateMeans(CANT_MEANS, items, CANT_ITERACIONES, size_lines, belongsTo, CANT_FEATURES);
    clusters = FindClusters(items, belongsTo, size_lines, CANT_MEANS, CANT_FEATURES);
    printf("Duración de CalculateMeans + FindClusters: %f seg\n", omp_get_wtime() - start2);
    
    printf("Valores de las medias finales:\n");
    for(int i = 0; i < CANT_MEANS; i++){
        printf("Mean[%d] -> (%lf,%lf)\n", i, means[i][0], means[i][1]);
    }

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

/**
 * @brief Creamos una lista de clusters, donde cada cluster es a su vez un arreglo que contiene 
 * todos los items que pertenecen a dicho cluster.
 */
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
        //printf("belong: %lu\n", belongsTo[i]);
    }
     printf("%lf %lf", clusters[1][3][0], clusters[1][3][1]);

    return clusters;
}

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
    u_int64_t cSize, index;

    //Inicializa las means (medias) con valores estimativos
    double** means = InitializeMeans(cant_means, cMin, cMax, cant_features);

    //Inicializa los clusters, clusterSizes almacena el numero de items de cada cluster
    u_int64_t* clusterSizes = calloc(cant_means, sizeof(u_int64_t));

    //Define un arreglo para almacenar el indice del cluster al que pertenece cada item
    //belongsTo = calloc(size_lines, sizeof(u_int64_t));

    //Calcula las medias
    for(j = 0; j < cant_iterations; j++) {
        
        //Si no ocurrio un cambio en el cluster, se detiene
        noChange = TRUE;
        countChangeItem = 0;

        //Resetea el clusterSizes a 0 para cada una de las medias
        memset(clusterSizes, 0, sizeof(u_int64_t)*cant_means);

        for(u_int64_t k = 0; k < size_lines; k++) { //se recorren todos los items
            item = items[k];

            //Clasifica item dentro de un cluster y actualiza las medias correspondientes
            index = Classify(means, item, cant_means, cant_features);

            clusterSizes[index] += 1;
            cSize = clusterSizes[index]; //cant de items del cluster
            //printf("Later - Mean[%lu] = %f\n", index,means[index]);
            //printf("Result:%f\n",updateMean(means[index], item, cSize));
            updateMean(means[index], item, cSize, cant_features);
            //printf("After - Mean[%lu] = %f\n", index,means[index]);

            //si el Item cambio de cluster
            if(index != belongsTo[k]){
                noChange = FALSE;
                countChangeItem++;
                belongsTo[k] = index;
            }

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
    for (int m = 0; m < cant_means; m++) {
        printf("Cluster[%d]: %lu\n", m, clusterSizes[m]);
    }
    printf("Cantidad de iteraciones: %d\n", j);

    free(clusterSizes);
    free(cMin);
    free(cMax);
    //free(item);
   // free(belongsTo);
    return means;
}

/**
 * @brief Clasifica un item dentro de una media (o cluster), de acuerdo a la distancia euclidiana.
 * @param means arreglo de medias
 * @param item item a clasificar
 * @param cant_means cantidad de medias o clusters 
 * @param cant_features cantidad de caracteristicas de cada item
 * @return el indice de la media a la que se asocio el item.
 */
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

/**
 * @brief Calcula distancia euclidiana entre dos vectores.
 * @param x primer vector (item)
 * @param y segundo vector (item)
 * @param length longitud del vector
 * @return distancia euclidiana entre ambos vectores.
 */
double distanciaEuclidiana(double* x, double* y, int length){
    double distancia = 0;
    for(int i = 0; i < length; i++){
        distancia += pow((x[i] - y[i]), 2);
    }
   
    return sqrt(distancia);
}

/*
Actualiza el valor de la media. 
- mean es la media que se va a cambiar,
- item es el nuevo valor que se quiere introducir en el cluster de esa media,
- cantItems es la cantidad de item en el cluster de esa media
mean e item son de 1 dimension (cada item tiene una sola feature)
Formula: m = (m*(n-1)+x)/n
no retorna nada porque el puntero se pasa por referencia
*/
/**
 * @brief Actualiza el valor de la media incorporando un nuevo item al cluster. 
 * @param mean es la media (arreglo) que se va a actualizar.
 * @param item es el nuevo valor que se quiere introducir en el cluster de esa media.
 * @param cant_items es la cantidad de item en el cluster de esa media.
 * @param cant_features cantidad de caracteristicas de un item.
 * Formula que se aplica para calcular la nueva media: m = (m*(n-1)+x)/n
 */
void updateMean(double* mean, double* item, u_int64_t cant_items, u_int8_t cant_features){
    double m;

    for(u_int8_t i = 0; i < cant_features; i++){
        m = mean[i];
        m = (m * ((double) cant_items - 1) + item[i]) / (double) cant_items;
        mean[i] = round(m); //se redondea a 3 cifras decimales
    } 
}

/**
 * @brief redondea un numero de punto flotante a 3 cifras despues de la coma.
 * @param var numero a redondear
 * @return numero con 3 cifras decimales
 * Ejemplo: 37.66666 * 1000 = 37666.66
 * 37666.66 + .5 = 37667.16    for rounding off value
 * then type cast to int so value is 37667
 * then divided by 1000 so the value converted into 37.67
 */
double round(double var){
    double value = (int)(var * 1000 + .5);
    return (double)value / 1000;
}

/**
 * @brief Lee el archivo indicado y carga el arreglo de items.
 * @param filename string nombre del archivo que contiene los datos
 * @param size_lines cantidad de lineas del archivo
 * @param cant_features cantidad de features de cada item (cantidad de columnas del archivo separadas por comas) 
 * @return arreglo doble con cantidad de filas igual a cantidad de items y cantidad de columnas igual a cantidad de features.
 */
double** ReadData(char filename[TAM_MAX_FILENAME], u_int64_t size_lines, u_int8_t cant_features){
    
    FILE *file = fopen(filename, "r");
    //u_int64_t size_lines = CalcLines(f);
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
              //  feature = strtod(token, &ptr); // 2do elemento
              //  items[i][j] = feature;
            }
            i++;
        }
    }
    free(line);
    fclose(file);

    return items;
}

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
/**
 * @brief Inicializa el arreglo de medias en valores equiespaciados en el rango de datos.
 * @param cant_means cantidad de medias o clusters
 * @param cMin vector con los valores minimos de cada feature
 * @param cMax vector con los valores maximos de cada feature
 * @param cant_features cantidad de features (o columnas) de cada item
 * @return arreglo con las medias (1 por cada cluster).
 * 
 */
double** InitializeMeans(u_int16_t cant_means, double* cMin, double* cMax, u_int8_t cant_features){
    
    double **means = (double **) malloc(cant_means * sizeof(double*));
    for(u_int16_t n = 0; n < cant_means; n++){
        means[n] = (double *) malloc(cant_features * sizeof(double));
    }
  //  double *range = (double *) malloc();
    //double range = cMax - cMin;
    //double jump = range / cant_means;

    //definimos el salto de un valor de media al siguiente
    double *jump = (double *) malloc(cant_features * sizeof(double));
    for(u_int8_t n = 0; n < cant_features; n++){
        jump[n] = (double) (cMax[n] - cMin[n]) / cant_means;
    }

   printf("Valores de las medias iniciales:\n");
    for(u_int16_t i = 0; i < cant_means; i++){
        for(u_int8_t j = 0; j < cant_features; j++){
            means[i][j] = cMin[j] + (0.5 + i) * jump[j];
        }
        printf("Mean[%d] -> (%lf,%lf)\n", i, means[i][0], means[i][1]);
    }

    free(jump);
    return means;
}

// PODRIAMOS HACER QUE CALCULE LA CANTIDAD DE FEATURES TAMBIEN
/**
 * @brief Cuenta la cantidad de lineas del archivo (para definir el tamaño del arreglo items posteriormente)
 * @param filename nombre del archivo
 * @return cantidad de lineas (o items) del archivo
 */
u_int64_t CalcLines(char filename[TAM_MAX_FILENAME]) {
    FILE *f = fopen(filename, "r");
    u_int64_t cant_lines = 0; 
    char* cadena = calloc(TAM_LINEA, sizeof(char));
    char* valor;
    while(fgets(cadena, TAM_LINEA, f)){
        valor = strstr(cadena, ",");
        valor++;
        //printf("valor: %s\n",valor);
        if(valor != NULL && strcmp(valor,"values\n") && strcmp(valor,"\n")){
            //printf("line:%s\n",cadena);
            cant_lines++;
        }
    }
    free (cadena);
    fclose(f);
    printf("Cantidad de items: %ld\n", cant_lines);
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

/**
 * @brief Encontramos minimo y maximo para cada feature del arreglo items.
 * @param items
 * @param size_lines
 * @param minimo
 * @param maximo
 * @param cant_features
 */
void searchMinMax(double** items, u_int64_t size_lines, double* minimo, double* maximo, u_int8_t cant_features){


    //Define el maximo como el minimo valor de tipo DOUBLE
   // double maximo = DBL_MIN;
    //Define el minimo como el maximo valor de tipo DOUBLE
   // double minimo = DBL_MAX;

   // double *maximo, *minimo;
   // maximo = (double*) malloc(cant_features * sizeof(double));
   // minimo = (double*) malloc(cant_features * sizeof(double));

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
   // *cMin = minimo;
   // *cMax = maximo;
}
