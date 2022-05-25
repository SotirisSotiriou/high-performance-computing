#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

__global__ void string_matching(char *buffer, char *pattern, int match_size, int pattern_size, int *match){
    int tid, i;

    for(tid=blockIdx.x*blockDim.x+threadIdx.x; tid<match_size; tid+=blockDim.x){
        for (i = 0; i < pattern_size && pattern[i] == buffer[i + tid]; ++i);
        if(i >= pattern_size){
            match[tid] = 1;
        }
        else{
            match[tid] = 0;
        }
    }

}


int main(int argc, char *argv[]){
    FILE *pFile;
    int i;
	long file_size, match_size, pattern_size;
	char * buffer;
	char * filename, *pattern;
	size_t result;
	int *match, total_matches;

    //CUDA variables
    int blocks, threads_per_block;
    int *match_dev;
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
	
	match = (int *) malloc (sizeof(int)*match_size);
	if (match == NULL) {printf ("Malloc error\n"); return 5;}

    cudaMalloc((void **)&match_dev, match_size*sizeof(int));
    cudaMalloc((void **)&buffer_dev, file_size*sizeof(char));
    cudaMalloc((void **)&pattern_dev, pattern_size*sizeof(char));

    cudaEventRecord(total_start);

    cudaEventRecord(comp_start);

    cudaMemcpy(buffer_dev, buffer, file_size*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(pattern_dev, pattern, pattern_size*sizeof(char), cudaMemcpyHostToDevice);

    string_matching<<<blocks, threads_per_block>>>(buffer_dev, pattern_dev, match_size, pattern_size, match_dev);
    cudaThreadSynchronize();

    cudaEventRecord(comp_stop);
    cudaEventSynchronize(comp_stop);
    cudaEventElapsedTime(&comp_time, comp_start, comp_stop);

    cudaMemcpy(match, match_dev, match_size*sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(total_stop);
    cudaEventSynchronize(total_stop);
    cudaEventElapsedTime(&total_time, total_start, total_stop);

    total_matches = 0;
    for(i=0; i<match_size; i++){
        total_matches += match[i];
    }

    cudaFree(match_dev);
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