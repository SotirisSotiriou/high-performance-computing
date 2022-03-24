/*
 * countsort_parallel_rand.c
 * 
 * Copyright 2022 Sotiris <sotiris@sotiris-lubuntu>
 * 
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void Count_sort(int *a, long n, int threads);
void readFromFile(char *filename, int *a, long size);
void writeOutput(char *filename, int *a, long size);
//void printTable(int *table, long size);


// ./<executable_file> <input_file> <size> <output_file> <num_of_threads>
int main(int argc, char **argv)
{
	if(argc != 3){
        printf("Using: %s <size> <num_of_threads>\n", argv[0]);
        exit(1);
    }

	long size = strtol(argv[1],NULL, 10);
	int num_threads = strtol(argv[2],NULL,10);
	
	//initialize data
    printf("Initializing data...\n");
	int *a = malloc(size*sizeof(int));
	long i;
    for(i=0; i<size; i++){
        a[i] = rand()%100;
    }
    printf("Initialization completed\n");
	
	//calculation
	Count_sort(a,size,num_threads);

	return 0;
}


void Count_sort(int *a, long n, int threads){
	long i, j, count;
	int *temp = malloc(n*sizeof(int));
	
	printf("Sorting...\n");
	
	double time = omp_get_wtime();
	
#pragma omp parallel for num_threads(threads) default(none) private(count,i,j) shared(a,temp,n)
	for(i=0; i<n; i++){
		count = 0;
		for(j=0; j<n; j++){
			if(a[j] < a[i]){
				count++;
			}
			else if(a[j] == a[i] && j < i){
				count++;
			}
		}
		temp[count] = a[i];
	}
	
	time = omp_get_wtime() - time;
	
	printf("Sort completed\n\n");
	
	memcpy(a, temp, n*sizeof(int));
	free(temp);

	//print result
    for(i=0; i<n; i++){
        printf("%d ", a[i]);
    }
	
	printf("\n\nparallel countsort with %d threads took %.5f seconds\n", threads, time);
}

