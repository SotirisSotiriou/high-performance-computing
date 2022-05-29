#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define UPPER 100
#define LOWER 0

__global__ void countsort(int *x, int *y, int N, int slice, int extra, int total_threads){
    int index = threadIdx.x + blockIdx.x * blockDim.x ;
    int start = index * slice; 
    int stop = start + slice;

    if (index == (total_threads-1))
    stop += extra;

    int i,j;
    int my_num, my_place;

    for(j=start; j<stop; j++){
        my_num = x[j];
        my_place = 0;
        for(i=0; i<N; i++){
            if ((my_num > x[i]) || ((my_num == x[i]) && (j < i))){
                my_place++;
            }
        }
        y[my_place] = my_num;
    }
}


int main(int argc, char *argv[]){
    int i, N, threads_per_block, total_blocks, total_threads, slice, extra;

    float total_time, comp_time;
    cudaEvent_t total_start, total_stop, comp_start, comp_stop;
    cudaEventCreate(&total_start);
  	cudaEventCreate(&total_stop);
  	cudaEventCreate(&comp_start);
  	cudaEventCreate(&comp_stop);

    //Διανύσματα στον host (CPU)
    int *x;


    if (argc != 4) {
        printf ("Usage : %s <vector_size> <blocks> <threads_per_block>\n", argv[0]);
        return 1;
    }

    N = strtol(argv[1], NULL, 10);
    total_blocks = strtol(argv[2], NULL, 10);
    threads_per_block = strtol(argv[3], NULL, 10);
    total_threads = total_blocks*threads_per_block;

    x = (int *) malloc(N*sizeof(int));

    //Αρχικοποίηση διανύσματος
    for(i=0; i<N; i++){
        //x[i] = N - i;
        x[i] = (rand() % (UPPER - LOWER + 1)) + LOWER;
    }

    //Αρχικός πίνακας
    if(N <= 100){
        printf("\n\n\nVector:\n");
        for (i=0; i<N; i++) {
            printf("%d ", x[i]);
        }
    }

    //Διανύσματα στο device (GPU)
    int *x_dev, *y_dev;
    
    cudaMalloc((void **)&x_dev, N*sizeof(int));
    cudaMalloc((void **)&y_dev, N*sizeof(int));

    cudaEventRecord(total_start);

    //Αντιγραφή δεδομένων από τον host (CPU) στο device (GPU)
    cudaMemcpy(x_dev, x, N*sizeof(int), cudaMemcpyHostToDevice);

    cudaEventRecord(comp_start);

    slice = N / total_threads;
    extra = N % total_threads;

    //Παράλληλη ταξινόμηση του x_dev
    countsort<<<total_blocks, threads_per_block>>>(x_dev,y_dev,N,slice,extra,total_threads);

    cudaEventRecord(comp_stop);
    cudaEventSynchronize(comp_stop);
    cudaEventElapsedTime(&comp_time, comp_start, comp_stop);

    //Αντιγραφή του αποτελέσματος από το device (GPU) στον host (CPU)
    cudaMemcpy(x, y_dev, N*sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(total_stop);
    cudaEventSynchronize(total_stop);
    cudaEventElapsedTime(&total_time, total_start, total_stop);

    //Απελευθέρωση της μνήμης που δεσμεύεται από τα x_dev, y_dev
    cudaFree(x_dev);
	cudaFree(y_dev);

    // Εκτύπωση αποτελέσματος από τον host (CPU)
    if(N <= 100){
        printf("\n\n\nCount sort result:\n");
        for (i=0; i<N; i++) {
            printf("%d ", x[i]);
        }
    }

    free(x);

    //GPU Timing
    printf("\n\n\nN: %d, Blocks: %d, Threads: %d\n", N, total_blocks, total_threads);
    printf("Total time (ms): %.3f\n", total_time);
    printf("Kernel time (ms): %.3f\n", comp_time);
    printf("Data transfer time(ms): %.3f\n\n\n", total_time-comp_time);   

    return 0;
}