/**
 * @file kmeans-mpi-naive.c
 * @brief Algoritmo Kmeans de clustering, se utiliza para agrupar items con caracteristicas similares.
 * @author Jeremias Agustinoy y Alejandro Ferrero
 * @version 1.0
 * @date 26/05/2021
 */

#include "kmeans-mpi-naive.h"
#include "mpi.h"

//#define NUM_PROCS 4  //numero de procesasdores


int main(int argc, char* argv[]) {
    double start, end, start2; 
    int rank, num_tasks;
    double ***clusters;
    u_int64_t *belongsTo;  //Define un arreglo para almacenar el indice del cluster al que pertenece cada item
    u_int64_t size_lines = 0;
    //double  **items;
    double **all_items;
    u_int64_t cant_items_proc; //cantidad de items por proceso
    double **means;
    u_int64_t resto = 0;

    MPI_Init(&argc, &argv);
    
    start = MPI_Wtime(); 

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
    printf("num tasks: %d\n", num_tasks);

    //el proceso con rango 0 es el que lee y distribuye los datos    
    if(rank == 0){ 
        start2 = MPI_Wtime(); 
        size_lines = CalcLines(PATH); //primero se calcula la cantidad de items
        all_items = ReadData(PATH, size_lines, CANT_FEATURES); // se leen los datos
        printf("Duración de CalcLines + ReadData en proceso %d: %f seg\n\n", rank, MPI_Wtime() - start2);
    }

    // se envia la cantidad total de items
    MPI_Bcast(&size_lines, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    printf("proc %d, size_lines: %lu\n", rank, size_lines);

//HACER PARA EL CASO QUE LA DIVISION NO SEA ENTERA(EL RESTO SE LO QUEDA EL PROC 0)
    cant_items_proc = size_lines/(unsigned)num_tasks;
    if(rank == 0){ //si la division no es entera y sobran items, los clasifica el proceso 0
        resto = size_lines % (unsigned)num_tasks;
    }
    printf("cant items proc: %ld\n", cant_items_proc);

    double **items = alloc_2d_double(cant_items_proc + resto, CANT_FEATURES); //resto puede ser distinto de cero para el proc 0.

    //se envia una cierta cantidad de items a cada proceso (se toman como resto los primeros items, van a parar al proceso 0)
    MPI_Scatter(&(all_items[0][0]) + resto*sizeof(double), (int)cant_items_proc, MPI_DOUBLE, &(items[0][0]) + resto*sizeof(double), (int)cant_items_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if(rank == 0){ //copio los items restantes al proc 0
        memcpy(&(items[0][0]), &(all_items[0][0]), resto*sizeof(double));
        cant_items_proc += resto; //se le agrega el resto al proceso 0
    }
  //  MPI_Scatter(&(all_items[0][0]), cant_items_proc, MPI_DOUBLE, &(items[0][0]), cant_items_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    printf("proc %d, items[0][0]: %lf\n", rank, items[0][0]);

    //Define un arreglo para almacenar el indice del cluster al que pertenece cada item
    belongsTo = calloc(cant_items_proc, sizeof(u_int64_t));
    printf("SE ENVIARON LOS ITEMS\n");

    //-----------------------------------
    double *cMin, *cMax;

    // el proceso de rango 0 es el encargado de inicializar los valores de las medias de cada cluster
    if(rank == 0){
        cMin = (double*) malloc(CANT_FEATURES * sizeof(double));
        cMax = (double*) malloc(CANT_FEATURES * sizeof(double));
        //Encuentra el minimo y maximo de cada columna (o feature)
        searchMinMax(all_items, size_lines, cMin, cMax, CANT_FEATURES);

        printf("MIN: %lf, MAX: %lf\n", cMin[0], cMax[0]);

        //Inicializa las means (medias) con valores estimativos
        means = (double **) InitializeMeans(CANT_MEANS, cMin, cMax, CANT_FEATURES);
        printf("MEAN: %lf, %lf\n", means[0][0], means[2][0]);

    }else{ //para el resto de procesos se asigna espacio para las medias
        means = (double **) alloc_2d_double(CANT_MEANS, CANT_FEATURES);
    }

    //-----------------------------------
    
    start2 = MPI_Wtime(); 
    means = CalculateMeans(CANT_MEANS, items, CANT_ITERACIONES, cant_items_proc, belongsTo, CANT_FEATURES, means);
    printf("Duración de CalculateMeans: %f seg\n", MPI_Wtime() - start2);

    /*if(rank == 0){
        start2 = MPI_Wtime();
        clusters = FindClusters(items, belongsTo, size_lines, CANT_MEANS, CANT_FEATURES);
        printf("Duración de FindClusters: %f seg\n", MPI_Wtime() - start2);
        
        printf("\nValores de las medias finales:\n");
        for(int i = 0; i < CANT_MEANS; i++){
            printf("Mean[%d] -> (%lf,%lf,%lf)\n", i, means[i][0], means[i][1], means[i][2]);
        }
    }

    //se libera la memoria del heap
    for(int n = 0; n < CANT_MEANS; n++){
        for(u_int64_t m = 0; m < size_lines; m++){
            free(clusters[n][m]);
        } 
        free(clusters[n]);
    }
    free(clusters);

    for(int n = 0; n < CANT_MEANS; n++){
        free(means[n]);
    }
    free(means);  
*/

    free(belongsTo);

    free(items[0]);
    free(items);

    free(means[0]);
    free(means);

    if(rank == 0){
        free(cMin);
        free(cMax);
        free(all_items[0]);
        free(all_items);
    }

    MPI_Finalize();

    end = MPI_Wtime(); 
    printf("\033[1;33m >>> Ejecución algoritmo K-means MPI <<<\033[0;37m \n");
    printf("Duración total del programa: %f seg\n", end - start);
    
    return EXIT_SUCCESS;
}

double **alloc_2d_double(u_int64_t rows, u_int64_t cols) {
    double *data = (double *)malloc(rows * cols * sizeof(double));
    double **array= (double **)malloc(rows * sizeof(double*));
    for (u_int64_t i = 0; i < rows; i++)
        array[i] = &(data[cols*i]);

    return array;
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

double** CalculateMeans(u_int16_t cant_means, double** items, int cant_iterations, u_int64_t cant_items_proc, u_int64_t* belongsTo, u_int8_t cant_features, double** means){
   // double *cMin, *cMax;
    int rank, num_tasks;
   // double **means;
    int noChange, all_noChange, j;
    double *item;
    u_int64_t index;
    u_int64_t countChangeItem, all_countChangeItem;
    double minPorcentaje;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);

    //define el porcentaje minimo de cambio de items entre clusters para que continue la ejecucion del algoritmo
    minPorcentaje = 0.001 * (double) cant_items_proc*num_tasks;
    
    //Inicializa los clusters, clusterSizes almacena el numero de items de cada cluster
    u_int64_t* clusterSizes = calloc(cant_means, sizeof(u_int64_t));
    u_int64_t* all_clusterSizes = calloc(cant_means, sizeof(u_int64_t));

    //guarda las suma de los valores de los items de cada cluster para despues calcular el promedio
    double sumas_items[cant_means][cant_features]; 
    double all_sumas_items[cant_means][cant_features];

    //Calcula las medias
    for(j = 0; j < cant_iterations; j++) {
        
        //Si no ocurrio un cambio en el cluster, se detiene
        noChange = TRUE;
        countChangeItem = 0;

        //Resetea el clusterSizes a 0 para cada una de las medias
        memset(clusterSizes, 0, sizeof(u_int64_t)*cant_means);
        memset(sumas_items, 0, sizeof(double)*cant_means*cant_features);

        printf("ANTES ENVIO MEANS\n");
        
        //envia las medias a los otros procesos
        MPI_Bcast(&(means[0][0]), cant_means*cant_features, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        for(u_int64_t k = 0; k < cant_items_proc; k++) { //cada proceso recorre sus items
            item = items[k];

            //Clasifica item dentro de un cluster y actualiza las medias correspondientes
            index = Classify(means, item, cant_means, cant_features);
           // printf("INDEX %d \n", index);
            clusterSizes[index] += 1;

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

        } //termina de clasificar a todos los items
        //se deben actualizar las medias globales de todos los procesos, para eso se hace una reduccion de las sumas y 
        //de la cantidad de items en cada cluster, y luego el proceso 0 calcular todas las medias globales
        printf("FIN ITERACION\n");

        MPI_Reduce(sumas_items, all_sumas_items, cant_means*cant_features, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(clusterSizes, all_clusterSizes, cant_means, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

        if(rank == 0){
            //calcula las nuevas medias dividiendo las sumas acumuladas por la cantidad de cada cluster
            for(int m = 0; m < cant_means; m++){
                if(all_clusterSizes[m] == 0) continue; //para evitar divisiones por cero, la media queda en el valor anterior

                for(int f = 0; f < cant_features; f++){
                    means[m][f] = all_sumas_items[m][f] / (double)all_clusterSizes[m];
                }
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

        //hace una AND de los noChange de todos los procesos (con que uno sea falso se debe dar una iteracion mas)
        MPI_Allreduce(&noChange, &all_noChange, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD); //reduce + broadcast

        //se suman la cantidad de items que cambiaron de todos los procesos
        MPI_Allreduce(&countChangeItem, &all_countChangeItem, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD); //reduce + broadcast

        if(all_noChange || (all_countChangeItem < minPorcentaje)){
            break;
        }
    }

    //envia las medias a los otros procesos
    MPI_Bcast(&(means[0][0]), cant_means*cant_features, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(rank == 0){
        printf("\n>>> Cantidad de items en cada cluster <<<\n");
        for (int m = 0; m < cant_means; m++) {
            printf("Cluster[%d]: %lu\n", m, all_clusterSizes[m]);
        }
        printf("Cantidad de iteraciones: %d\n", j);
    }

    free(clusterSizes);
    free(all_clusterSizes);

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

    /*double** items = (double **) malloc(size_lines * sizeof(double*));
    
    for(u_int64_t n = 0; n < size_lines; n++){
        items[n] = (double *) malloc(cant_features * sizeof(double));
    }*/

    //Definimos un arreglo de arreglos (cada item consta de 2 o mas features)
    double** items = (double **) alloc_2d_double(size_lines, cant_features);

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
    
    /*double **means = (double **) malloc(cant_means * sizeof(double*));
    for(u_int16_t n = 0; n < cant_means; n++){
        means[n] = (double *) malloc(cant_features * sizeof(double));
    }*/
    double **means = (double **) alloc_2d_double(cant_means, cant_features);

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
