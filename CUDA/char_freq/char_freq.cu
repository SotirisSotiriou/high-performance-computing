#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 128
#define base 0

__global__ void char_freq(char *buffer, int *freq, int buffersize, int blocks, int slice, int extra){
    int i, j;
    int thread_index = blockIdx.x*blockDim.x + threadIdx.x;
    int start = thread_index*slice;
    int stop = start + slice;
    if(thread_index == blocks*blockDim.x - 1){
        stop += extra;
    }
    if(stop > buffersize){
        stop = buffersize;
    }
    int temp[N];

    //Cyclic calculation of block local frequences
    for(i=start; i<stop; i++){
        temp[buffer[i]-base]++; //used for thread synchronization
    }

    //reduce temp results to freq
    for(j=0; j<N; j++){
        atomicAdd(&freq[j], temp[j]);
    }
}


int main(int argc, char *argv[]){
    FILE *pFile;
	long file_size;
	char * buffer;
	char * filename;
	size_t result;
	int j, freq[N];
    int slice, extra;
    int total_blocks, threads_per_block, total_threads;

    float total_time, comp_time;
    cudaEvent_t total_start, total_stop, comp_start, comp_stop;
    cudaEventCreate(&total_start);
  	cudaEventCreate(&total_stop);
  	cudaEventCreate(&comp_start);
  	cudaEventCreate(&comp_stop);


    if (argc != 4) {
		printf ("Usage : %s <file_name> <blocks> <threads_per_block>\n", argv[0]);
		return 1;
    }

    total_blocks = strtol(argv[2], NULL, 10);
    threads_per_block = strtol(argv[3], NULL, 10);
    total_threads = total_blocks*threads_per_block;

    filename = argv[1];
	pFile = fopen ( filename , "rb" );
	if (pFile==NULL) {printf ("File error\n"); return 2;}

    //obtain file size
	fseek (pFile , 0 , SEEK_END);
	file_size = ftell (pFile);
	rewind (pFile);
	printf("file size is %ld\n", file_size);
	
	//allocate memory to contain the file
	buffer = (char*) malloc (sizeof(char)*file_size);
	if (buffer == NULL) {printf ("Memory error\n"); return 3;}

	//copy file data to buffer 
	result = fread (buffer,1,file_size,pFile);
	if (result != file_size) {printf ("Reading error\n"); return 4;}

    //Device arrays (GPU)
    char *buffer_dev;
    int *freq_dev;

    cudaMalloc((void **)&buffer_dev, file_size*sizeof(char));
    cudaMalloc((void **)&freq_dev, N*sizeof(int));
    cudaMemset(freq_dev,0,N*sizeof(int));

    cudaEventRecord(total_start);

    //Copy data from host (CPU) to device (GPU)
    cudaMemcpy(buffer_dev, buffer, file_size*sizeof(char), cudaMemcpyHostToDevice);

    cudaEventRecord(comp_start);

    total_threads = total_blocks*threads_per_block;
    slice = file_size/total_threads;
    extra = file_size%total_threads;

    char_freq<<<total_blocks, threads_per_block>>>(buffer_dev, freq_dev, file_size, total_blocks, slice, extra);

    cudaEventRecord(comp_stop);
    cudaEventSynchronize(comp_stop);
    cudaEventElapsedTime(&comp_time, comp_start, comp_stop);

    //Copy result from device (GPU) to host (CPU)
    cudaMemcpy(freq, freq_dev, N*sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(total_stop);
    cudaEventSynchronize(total_stop);
    cudaEventElapsedTime(&total_time, total_start, total_stop);

    cudaFree(buffer_dev);
    cudaFree(freq_dev);

    //Print Result
    for (j=0; j<N; j++){
		printf("%d = %d\n", j+base, freq[j]);
	}
    
    fclose (pFile);
	free (buffer);

    //GPU Timing
    printf("\n\n\nN: %d, Blocks: %d, Threads: %d\n", N, total_blocks, total_threads);
    printf("Total time (ms): %.3f\n", total_time);
    printf("Kernel time (ms): %.3f\n", comp_time);
    printf("Data transfer time(ms): %.3f\n\n\n", total_time-comp_time);

    return 0;
}