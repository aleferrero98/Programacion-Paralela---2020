#include "stdio.h"
#include "float.h"
#include <cuda.h>

#define HILOS 128
#define PATH "./inputs/randomData_2M_3feature.csv"
#define CANT_FEATURES 3
#define CANT_MEANS 4
#define CANT_ITERACIONES 100
#define MAX_DOUBLE DBL_MAX

//Funciones CUDA
__global__ void kMeansClusterAssignment(double* means_dev, double* items_dev, int *clusterAsignado_dev,int *countChangeItem_dev );
__global__ void kMeansCentroidUpdate(double *items_dev, int *clusterAsignado_dev, double *means_dev, int *d_clust_sizes);
__device__ u_int64_t Classify(double* means_dev, double* item, int cant_means, int cant_features);
__device__ double distanciaEuclidiana(double* x , double* y, int length);

//Funciones HOST
double** CalculateMeans(double* items_dev, double** means, u_int64_t size_lines, int *clusterAsignado_dev, int nBloques, int hilosB);
double*** FindClusters(int *clusterAsignado_dev, u_int64_t cant_items, double **items);
u_int64_t CalcLines(char filename[50]);
double **alloc_2d_double(u_int64_t rows, u_int64_t cols);
double** ReadData(char filename[50], u_int64_t size_lines, u_int8_t cant_features);
void searchMinMax(double** items, u_int64_t size_lines, double* minimo, double* maximo, u_int8_t cant_features);
double** InitializeMeans(u_int16_t cant_means, double* cMin, double* cMax, u_int8_t cant_features);
__host__ void check_CUDA_Error(const char *mensaje);

//Constantes de CUDA
__constant__ u_int64_t CANT_ITEMS_CUDA;

int main()
{
    //Declaracion de eventos para tomar tiempos
    cudaEvent_t start;
    cudaEvent_t stop;

    //Creacion de eventos
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //Marca de inicio CalcLines y ReadData
    cudaEventRecord(start,0);
    //Calcula la cantidad de lineas del CSV
    u_int64_t size_lines = CalcLines(PATH);

    cudaMemcpyToSymbol(CANT_ITEMS_CUDA, &size_lines, sizeof(u_int64_t));
    check_CUDA_Error("ERROR en cudaMemcpyToSymbol");

    // double maxDouble = DBL_MAX;
    // cudaMemcpyToSymbol(MAX_DOUBLE, &maxDouble, sizeof(double));
    // check_CUDA_Error("ERROR en cudaMemcpyToSymbol");

    double **items = ReadData(PATH, size_lines, CANT_FEATURES);
    //Marca de final CalcLines y ReadData
    cudaEventRecord(stop,0);
    //Sincronizacion GPU-CPU
    cudaEventSynchronize(stop);
    //Calculo del tiempo en milisegundos
    float elapsedTime2;
    cudaEventElapsedTime(&elapsedTime2,start,stop);

    //Marca de inicio SearchMinMax, Calculo de hilos-bloques CUDA e Inicializacion Medias
    cudaEventRecord(start,0);
    double *cMin, *cMax;
    cMin = (double*) malloc(CANT_FEATURES * sizeof(double));
    cMax = (double*) malloc(CANT_FEATURES * sizeof(double));
    //Encuentra el minimo y maximo de cada columna (o feature)
    searchMinMax(items, size_lines, cMin, cMax, CANT_FEATURES);

    printf("MIN: %lf, MAX: %lf\n", cMin[0], cMax[0]);
    
    // calculamos el numero de bloques necesario para un tamaño de bloque fijo
    int nBloques = size_lines/HILOS;
    if (size_lines%HILOS != 0)
    {
        nBloques = nBloques + 1;
    }
    int hilosB = HILOS;

    //Inicializa las means (medias) con valores estimativos
    double** means = InitializeMeans(CANT_MEANS, cMin, cMax, CANT_FEATURES);

    //Marca de final SearchMinMax, Calculo de hilos-bloques CUDA e Inicializacion Medias
    cudaEventRecord(stop,0);
    //Sincronizacion GPU-CPU
    cudaEventSynchronize(stop);
    //Calculo del tiempo en milisegundos
    float elapsedTime3;
    cudaEventElapsedTime(&elapsedTime3,start,stop);

    //Almacena los indices de los items
    int *clusterAsignado_dev = 0;
    cudaMalloc(&clusterAsignado_dev,size_lines*sizeof(int));
    cudaMemset(clusterAsignado_dev,0,size_lines*sizeof(int));

    double* items_dev;
    cudaMalloc( (void**)&items_dev, size_lines*CANT_FEATURES*sizeof(double));
    check_CUDA_Error("ERROR en cudaMalloc");
    cudaMemcpy( items_dev, &items[0][0], size_lines*CANT_FEATURES*sizeof(double), cudaMemcpyHostToDevice );
    check_CUDA_Error("ERROR en cudaMemcpy items_dev");
    
    //Marca de inicio CalculateMeans
    cudaEventRecord(start,0);
    //Funcion que calcula las medias nuevas
    means = CalculateMeans(items_dev, means, size_lines, clusterAsignado_dev ,nBloques, hilosB);
    //Marca de final CalculateMeans
    cudaEventRecord(stop,0);
    //Sincronizacion GPU-CPU
    cudaEventSynchronize(stop);
    //Calculo del tiempo en milisegundos
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime,start,stop);

    //Marca de inicio FindCluster
    cudaEventRecord(start,0);
    //Funcion que calcula las medias nuevas
    double ***clusters = FindClusters(clusterAsignado_dev, size_lines, items);
    //Marca de final CalculateMeans
    cudaEventRecord(stop,0);
    //Sincronizacion GPU-CPU
    cudaEventSynchronize(stop);
    //Calculo del tiempo en milisegundos
    float elapsedTime4;
    cudaEventElapsedTime(&elapsedTime4,start,stop);

    //Liberacion de recursos
    for(int n = 0; n < CANT_MEANS; n++){
        for(u_int64_t m = 0; m < size_lines; m++){
            free(clusters[n][m]);
        } 
        free(clusters[n]);
    }
    free(clusters);
    free(items[0]);
    free(items);
    free(means[0]);
    free(means);
    free(cMin);
    free(cMax);
    cudaFree(clusterAsignado_dev);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    //Impresion de resultados
    printf("> Tiempo de ejecucion de CalcLines y ReadData: %f ms\n",elapsedTime2);
    printf("> Tiempo de ejecucion de SearchMinMax, Calculo de hilos-bloques CUDA e Inicializacion Medias: %f ms\n",elapsedTime3);
    printf("> Tiempo de ejecucion de CalculateMeans: %f ms\n",elapsedTime);
    printf("> Tiempo de ejecucion de FindCluster: %f ms\n",elapsedTime4);
    printf("> Tiempo de total del programa: %f ms\n", elapsedTime + elapsedTime2 + elapsedTime3 + elapsedTime4);

    return EXIT_SUCCESS;
}
/**
 * @brief Funcion que se encarga de armar una matriz 3D, donde se insertaran los items de acuerdo a su clasificacion
 * @param clusterAsignado_dev Arreglo 1D del cluster a que corresponde cada item
 * @param cant_items Cantidad de items
 * @param items Items a clasificar
 * @return Arreglo 3D de Clusters finales de acuerdo a la clasificacion de los items en cada media
 */
double*** FindClusters(int *clusterAsignado_dev, u_int64_t cant_items, double **items)
{
    // clusters es un array de 3 dimensiones, es un conjunto de clusters.
    // cada cluster es un conjunto de items.
    // cada item es un conjunto de features.
    double ***clusters = (double ***) malloc(CANT_MEANS * sizeof(double**));

    //Inicializa clusters
    for(u_int8_t n = 0; n < CANT_MEANS; n++){
        clusters[n] = (double **) malloc(cant_items * sizeof(double*));
        for(u_int64_t m = 0; m < cant_items; m++){
            clusters[n][m] = (double *) malloc(CANT_FEATURES * sizeof(double));
        }
    }

    int *clusterAsignado = (int*)malloc(cant_items*sizeof(int));
    cudaMemcpy(clusterAsignado, clusterAsignado_dev, cant_items*sizeof(int), cudaMemcpyDeviceToHost );
    int indices_series[CANT_MEANS]; 
    memset(indices_series, 0, sizeof(int)*CANT_MEANS);
    for(u_int64_t i = 0; i < cant_items; i++){
        for(u_int8_t j = 0; j < CANT_FEATURES; j++){ //se cargan todas las features del item al cluster
            clusters[clusterAsignado[i]][indices_series[clusterAsignado[i]]][j] = items[i][j];
        }
        indices_series[clusterAsignado[i]]++;
    }
    return clusters;
}
/**
 * @brief Funcion que se encarga de clasificar los items en las medias correspondientes
 * @param items_dev Items a clasificar, cada item contiene un valor por Feature, representada como arreglo 1D
 * @param means_dev Matriz de medias (Cantidad de Features * Cantidad de Medias), representada como arreglo 1D
 * @param size_lines Cantidad de items
 * @param clusterAsignado_dev Arreglo 1D del cluster a que corresponde cada item
 * @param nBloques Cantidad de bloques CUDA
 * @param hilosB Cantidad de hilos CUDA
 * @return Arreglo 2D de Medias finales de acuerdo a la clasificacion de los items
 */
double** CalculateMeans(double* items_dev, double** means, u_int64_t size_lines, int *clusterAsignado_dev, int nBloques, int hilosB)
{
    double minPorcentaje;
    //define el porcentaje minimo de cambio de items entre clusters para que continue la ejecucion del algoritmo
    minPorcentaje = 0.001 * (double) size_lines;

    printf("Porentaje minimo = %.2lf\n", minPorcentaje);

    double* means_dev;
    cudaMalloc( (void**)&means_dev, CANT_MEANS*CANT_FEATURES*sizeof(double));
    check_CUDA_Error("ERROR en cudaMalloc");

    /*Arreglo de cluster sizes*/
    //Creo y reseteo a 0 la variable de host
    int *h_clust_sizes = (int*)malloc(CANT_MEANS*sizeof(int));
    memset(h_clust_sizes, 0, sizeof(int)*CANT_MEANS);
    //cudaMemset(countChangeItem_dev, 0, sizeof(int));
    //Creo la variable de device
    int *d_clust_sizes = 0;
    cudaMalloc(&d_clust_sizes,CANT_MEANS*sizeof(float));
    check_CUDA_Error("ERROR en cudaMalloc d_clust_sizes ");
    //Copio lo que hay en host a device
    cudaMemcpy(d_clust_sizes,h_clust_sizes,CANT_MEANS*sizeof(int),cudaMemcpyHostToDevice);
    check_CUDA_Error("ERROR en cudaMemcpy d_clust_sizes ");
    //Almacena contador de cambios de items
    int *countChangeItem_dev = 0;
    cudaMalloc(&countChangeItem_dev,sizeof(int));
    
    int *countChangeItem = (int*)malloc(sizeof(int));

    //Calcula las medias
    for(int j = 0; j < CANT_ITERACIONES; j++) {
        
        printf("Iteracion: %d\n", j);

        //En cada iteracion, cantidad de cambios es 0
        //memset(countChangeItem, 0, sizeof(int));
                
        //Paso lo que hay en means a la placa luego de cambiarlo
        cudaMemcpy( means_dev, &means[0][0], CANT_MEANS*CANT_FEATURES*sizeof(double), cudaMemcpyHostToDevice );
        check_CUDA_Error("ERROR en cudaMemcpy means_dev");

        //Reseteo la cantidad de elementos de cada media en cada iteracion
        cudaMemset(d_clust_sizes,0,CANT_MEANS*sizeof(int));
        check_CUDA_Error("ERROR en cudaMemset means_dev");

        kMeansClusterAssignment<<<nBloques,hilosB>>>(items_dev, means_dev, clusterAsignado_dev, countChangeItem_dev);

        //Copio las nuevas medias obtenidas en la placa a las medias de Host
        cudaMemcpy(countChangeItem,countChangeItem_dev,sizeof(int),cudaMemcpyDeviceToHost);

        //Reseteo means para la placa, ya que se va a cambiar
        cudaMemset(means_dev,0,CANT_MEANS*CANT_FEATURES*sizeof(double));
        check_CUDA_Error("ERROR en cudaMemset means_dev");

        kMeansCentroidUpdate<<<nBloques,hilosB>>>(items_dev,clusterAsignado_dev,means_dev,d_clust_sizes);

        //Copio las nuevas medias obtenidas en la placa a las medias de Host
        cudaMemcpy(&means[0][0],means_dev,CANT_MEANS*CANT_FEATURES*sizeof(double),cudaMemcpyDeviceToHost);
        check_CUDA_Error("ERROR en cudaMemcpy means_dev 3");
        //Copio la cantidad de items de cada medias obtenidas en la placa al arreglo del host
        cudaMemcpy(h_clust_sizes, d_clust_sizes, CANT_MEANS*sizeof(int), cudaMemcpyDeviceToHost );
        check_CUDA_Error("ERROR en cudaMemcpy h_clust_sizes ");

        for (int a = 0; a < CANT_MEANS; a++)
        {
            for(int b=0; b < CANT_FEATURES; b++)
            {
                //Asigno el nuevo valor de las medias sacando promedio
                means[a][b] = means[a][b] / h_clust_sizes[a];
            }
            printf("Mean[%d] -> (%lf,%lf,%lf)\n", a, means[a][0], means[a][1],  means[a][2]);
            printf("Cluster[%d] -> %d\n", a, h_clust_sizes[a]);
        }
        
        //Comparo la cantidad de items cambiado en la iteracion actual con la anterior y si es menor al porcentaje
        //se deja de iterar
        printf("Cant cambios: %d\n",*countChangeItem);
        if(*countChangeItem < minPorcentaje){break;}
        //Reseteo cantidad de camios para la placa, ya que se va a cambiar
        cudaMemset(countChangeItem_dev,0,sizeof(int));
    }

    cudaFree(items_dev);
    cudaFree(means_dev);
    cudaFree(d_clust_sizes);
    free(h_clust_sizes);
    cudaFree(countChangeItem_dev);
    free(countChangeItem);
    return means;
}
/**
 * @brief Funcion que se encarga de obtener las sumas en cada media y la cantidad de elementos
 * @param items_dev Items a clasificar, cada item contiene un valor por Feature, representada como arreglo 1D
 * @param clusterAsignado_dev Arreglo 1D del cluster a que corresponde cada item
 * @param means_dev Matriz de medias (Cantidad de Features * Cantidad de Medias), representada como arreglo 1D
 * @param d_clust_sizes Arreglo 1D de la cantidad de items de cada media del cluster
 */
__global__ void kMeansCentroidUpdate(double *items_dev, int *clusterAsignado_dev, double *means_dev, int *d_clust_sizes)
{

	//Obtengo el ID de cada hilo
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;

	//Elimino aquellos que no deban trabajar
	if (idx >= CANT_ITEMS_CUDA) return;

	//Obtengo el ID de los hilos a nivel de bloque
	const int s_idx = threadIdx.x;

    //Armo un arreglo de items para cada bloque en memoria compartida
    __shared__ double items_bloque[HILOS][CANT_FEATURES];

    for(int i = 0; i < CANT_FEATURES; i++){
        items_bloque[s_idx][i] = items_dev[idx*CANT_FEATURES + i];
    }

    //Armo un arreglo de los cluster asignados para cada bloque en memoria compartida
	__shared__ int clusterAsignado_bloque[HILOS];
    clusterAsignado_bloque[s_idx] = clusterAsignado_dev[idx];

	__syncthreads();

    //Si es el hilo 0 de cada bloque, entonces suma los valores dentro de los arreglo compartido
	if(s_idx==0)
	{
        int limite = ((idx + blockDim.x) < CANT_ITEMS_CUDA)? blockDim.x : (CANT_ITEMS_CUDA - idx);

        //Creo arreglos de suma de valores del cluster del bloque y la cantidad de items de cada media
		double clust_sums[CANT_MEANS][CANT_FEATURES]={{0},{0},{0},{0}};
        int clust_sizes[CANT_MEANS]={0};

        //Se recorre el bloque, incrementando el cluster sizes de acuerdo a la media asignada y lo sumo 
		for(int j=0; j < limite; ++j)
		{
            int clust_id = clusterAsignado_bloque[j];
            clust_sizes[clust_id]+=1;
            for(int k = 0; k < CANT_FEATURES; ++k)
            {
                clust_sums[clust_id][k]+=items_bloque[j][k];
            }
		}

        //Por ultimo agregamos de forma atomica al arreglo means_dev la suma de todos los items designados en cada cluster
        //y al arreglo d_clust_sizes la cantidad de items en cada media
        int indice;
		for(int z=0; z < CANT_MEANS; ++z)
		{
            indice = z*CANT_FEATURES;
            for(int s=0; s < CANT_FEATURES ; s++)
            {
                atomicAdd(&means_dev[indice+s],clust_sums[z][s]);
            }
            atomicAdd(&d_clust_sizes[z],clust_sizes[z]);
        }
	}

	__syncthreads();
}
/**
 * @brief Funcion que se encarga de asignar los indices de cluster a cada item 
 * @param items_dev Items a clasificar, cada item contiene un valor por Feature, representada como arreglo 1D
 * @param means_dev Matriz de medias (Cantidad de Features * Cantidad de Medias), representada como arreglo 1D
 * @param clusterAsignado_dev Arreglo 1D del cluster a que corresponde cada item
 */
__global__ void kMeansClusterAssignment(double *items_dev, double *means_dev, int *clusterAsignado_dev,int *countChangeItem_dev )
{
    
    //Obtengo el ID para cada hilo
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;

    //Descarto aquellos hilos que no deban trabajar
	if (idx >= CANT_ITEMS_CUDA) return;
    
    //Obtengo el item correspondiente a cada hilo
    double *item = &items_dev[idx*CANT_FEATURES];

    u_int64_t index = Classify(means_dev, item, CANT_MEANS, CANT_FEATURES);

    if(clusterAsignado_dev[idx] != (int)index)
    {
        atomicAdd(countChangeItem_dev,1);
    }
    //Asigno cada item en un cluster y almaceno el indice de clasificacion en un arreglo
	clusterAsignado_dev[idx]=(int)index;
}
/**
 * @brief Funcion que se encarga de obtener el indice del cluster al que pertenece el item
 * @param means_dev Matriz de medias (Cantidad de Features * Cantidad de Medias), representada como arreglo 1D
 * @param item Item a clasificar
 * @param cant_means Cantidad de Medias
 * @param cant_features Cantidad de Features
 * @return Indice del cluster al que corresponde el Item
 */
__device__ u_int64_t Classify(double* means_dev, double* item, int cant_means, int cant_features){
    double minimun = MAX_DOUBLE;
    int index = -1;
    double distance;

    for(int i = 0; i < cant_means; i++){
        //calcula la distancia de un item a la media
        //printf("Means_dev: %ld\n", means_dev[i*3]);
        distance = distanciaEuclidiana(item, &means_dev[i*cant_features], cant_features);
        if(distance < minimun){
            minimun = distance;
            index = i;
        }
    }
    return (u_int64_t) index;
}
/**
 * @brief Funcion que se encarga de calcular la distancia Euclideana entre el item y las distintas Medias (2 vectores)
 * @param x Item (Vector 1)
 * @param y Medias (Vector 2)
 * @param length longitud del vector (Cantidad de Features)
 * @return Distancia euclidiana entre ambos vectores.
 */
__device__ double distanciaEuclidiana(double* x , double* y, int length){
    double distancia = 0;
    for(int i = 0; i < length; i++){
        distancia += pow((x[i] - y[i]), 2);
    }
    return sqrt(distancia);
}
/**
 * @brief Funcion que se encarga de calcular la cantidad de items a clasificar
 * @param filename nombre del archivo
 * @return cantidad de lineas (o items) del archivo
 */
u_int64_t CalcLines(char filename[50]) {
    printf(filename);
    FILE *f = fopen(filename, "r");
    u_int64_t cant_lines = 0; 
    char* cadena = (char*) calloc(100, sizeof(char));
    char* valor;
    while(fgets(cadena, 100, f)){
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
/**
 * @brief Funcion que se encarga allocar una matriz 2D
 * @param rows filas de la matriz
 * @param cols columnas de la matriz
 * @return Matriz 2D
 */
double **alloc_2d_double(u_int64_t rows, u_int64_t cols) {
    double *data = (double *)malloc(rows * cols * sizeof(double));
    double **array= (double **)malloc(rows * sizeof(double*));
    for (u_int64_t i = 0; i < rows; i++)
        array[i] = &(data[cols*i]);

    return array;
}
/**
 * @brief Busca el minimo y maximo valor para cada feature del arreglo items.
 * @param items datos a clasificar
 * @param size_lines cantidad de items
 * @param minimo arreglo de los valores minimos de cada feature
 * @param maximo arreglo de los valores maximos de cada feature
 * @param cant_features cantidad de caracteristicas que tiene cada item
 */
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
/**
 * @brief Lee el archivo indicado y carga el arreglo de items.
 * @param filename string nombre del archivo que contiene los datos
 * @param size_lines cantidad de lineas del archivo
 * @param cant_features cantidad de features de cada item (cantidad de columnas del archivo separadas por comas) 
 * @return arreglo doble con cantidad de filas igual a cantidad de items y cantidad de columnas igual a cantidad de features.
 */
double** ReadData(char filename[50], u_int64_t size_lines, u_int8_t cant_features){
    
    FILE *file = fopen(filename, "r");
    rewind(file);

    //Definimos un arreglo de arreglos (cada item consta de 2 o mas features)
    double** items = (double **) alloc_2d_double(size_lines, cant_features);

    char* line = (char*)calloc(100, sizeof(char));
    double feature;
    u_int64_t i = 0, j = 0;
    char* ptr;

    while(fgets(line, 100, file)){
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
/**
 * @brief Funcion que se encarga de detectar error de CUDA 
 * @param mensaje Mensaje de error CUDA
 */
__host__ void check_CUDA_Error(const char *mensaje)
{
    cudaError_t error;
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        printf("ERROR %d: %s (%s)\n", error, cudaGetErrorString(error), mensaje);
        printf("\npulsa INTRO para finalizar...");
        fflush(stdin);
        char tecla = getchar();
        exit(-1);
    }
}
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
double** InitializeMeans(u_int16_t cant_means, double* cMin, double* cMax, u_int8_t cant_features){
    /*        |__Feature 0__|__Feature 1__|__Feature 2__|                
        Media0|_____________|_____________|_____________|
        Media1|_____________|_____________|_____________|
    */
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