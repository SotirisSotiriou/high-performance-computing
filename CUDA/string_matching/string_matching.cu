#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

#define MAX_THREADS_PER_BLOCK 100

__global__ void string_matching(char *buffer, char *pattern, int match_size, int pattern_size, int blocks, int slice, int extra, int *gout){
    int tid, i;
    int thread_index = blockIdx.x*blockDim.x + threadIdx.x;
    int start = thread_index*slice;
    int stop = start + slice;
    if(thread_index == blocks*blockDim.x - 1){
        stop += extra;
    }
    if(stop > match_size){
        stop = match_size;
    }
    __shared__ int r[MAX_THREADS_PER_BLOCK];
    int sum = 0;

    for(tid=start; tid<stop; tid++){
        for (i = 0; i < pattern_size && pattern[i] == buffer[i + tid]; ++i);
        if(i >= pattern_size){
            sum++;
        }
    }

    r[threadIdx.x] = sum;

    __syncthreads();
    
    //for debugging
    //printf("Block: %d, Thread: %d, Global Thread: %d, Start: %d, Stop: %d, Matches: %d, Block Matches: %d\n", blockIdx.x, threadIdx.x, thread_index, start, stop, r[threadIdx.x], r[0]);

    //works only for power of 2 threads_per_block
    //example image url: https://i.stack.imgur.com/jjQvK.png
    // for (int size = blockDim.x/2; size>0; size/=2) { //uniform
    //     if (threadIdx.x<size)
    //         r[threadIdx.x] += r[threadIdx.x+size];
    //     __syncthreads();
    // }

    //adds the next thread's result to the previous and all reduces to 0 thread
    //example image url: https://i.stack.imgur.com/9s8NN.png
    for(int size = blockDim.x-2; size>=0; size--){
        if(threadIdx.x == size){
            r[threadIdx.x] += r[threadIdx.x+1];
        }
        __syncthreads();
    }

    //for debugging
    // if(threadIdx.x == 0){
    //     printf("Block %d matches: %d\n", blockIdx.x, r[0]);
    // }


    if(threadIdx.x == 0){
        gout[blockIdx.x] = r[0];
    }

}


int main(int argc, char *argv[]){
    int i;
    FILE *pFile;
	long file_size, match_size, pattern_size;
	char * buffer;
	char * filename, *pattern;
	size_t result;
    int *results;
	int total_matches;

    //CUDA variables
    int blocks, threads_per_block, total_threads, slice, extra;
    int *results_dev;
    char *buffer_dev, *pattern_dev;

    float total_time, comp_time;
    cudaEvent_t total_start, total_stop, comp_start, comp_stop;
    cudaEventCreate(&total_start);
  	cudaEventCreate(&total_stop);
  	cudaEventCreate(&comp_start);
  	cudaEventCreate(&comp_stop);

    if (argc != 5) {
        printf ("Usage : %s <file_name> <string> <blocks> <threads_per_block>\n", argv[0]);
        return 1;
    }
	filename = argv[1];
	pattern = argv[2];
    blocks = strtol(argv[3], NULL, 10);
    threads_per_block = strtol(argv[4], NULL, 10);
	
	pFile = fopen ( filename , "rb" );
	if (pFile==NULL) {printf ("File error\n"); return 2;}

	// obtain file size:
	fseek (pFile , 0 , SEEK_END);
	file_size = ftell (pFile);
	rewind (pFile);
	printf("file size is %ld\n", file_size);
	
	// allocate memory to contain the file:
	buffer = (char*) malloc (sizeof(char)*file_size);
	if (buffer == NULL) {printf ("Memory error\n"); return 3;}

	// copy the file into the buffer:
	result = fread (buffer,1,file_size,pFile);
	if (result != file_size) {printf ("Reading error\n"); return 4;} 
	
	pattern_size = strlen(pattern);
	match_size = file_size - pattern_size + 1;

    results = (int *)malloc(blocks*sizeof(int));

    cudaMalloc((void **)&results_dev, blocks*sizeof(int));
    cudaMalloc((void **)&buffer_dev, file_size*sizeof(char));
    cudaMalloc((void **)&pattern_dev, pattern_size*sizeof(char));

    cudaEventRecord(total_start);

    cudaEventRecord(comp_start);

    cudaMemcpy(buffer_dev, buffer, file_size*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(pattern_dev, pattern, pattern_size*sizeof(char), cudaMemcpyHostToDevice);

    total_threads = blocks*threads_per_block;
    slice = match_size/total_threads;
    extra = match_size%total_threads;

    string_matching<<<blocks, threads_per_block>>>(buffer_dev, pattern_dev, match_size, pattern_size, blocks, slice, extra, results_dev);

    cudaEventRecord(comp_stop);
    cudaEventSynchronize(comp_stop);
    cudaEventElapsedTime(&comp_time, comp_start, comp_stop);

    cudaMemcpy(results, results_dev, blocks*sizeof(int), cudaMemcpyDeviceToHost);

    total_matches = 0;
    for(i=0; i<blocks; i++){
        total_matches += results[i];
    }

    cudaEventRecord(total_stop);
    cudaEventSynchronize(total_stop);
    cudaEventElapsedTime(&total_time, total_start, total_stop);

    cudaFree(results_dev);
    cudaFree(buffer_dev);
    cudaFree(pattern_dev);

    fclose (pFile);
	free (buffer);

    //Print result
    printf("Total matches: %d\n", total_matches);

    printf("\n\n\nN: %d, Blocks: %d, Threads: %d\n", file_size, blocks, blocks*threads_per_block);
    printf("Total time (ms): %.3f\n", total_time);
    printf("Kernel time (ms): %.3f\n", comp_time);
    printf("Data transfer time(ms): %.3f\n\n\n", total_time-comp_time);

}