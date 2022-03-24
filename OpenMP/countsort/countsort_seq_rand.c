/*
 * countsort_parallel_rand.c
 * 
 * Copyright 2022 Sotiris <sotiris@sotiris-lubuntu>
 * 
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void Count_sort(int *a, long n);
void readFromFile(char *filename, int *a, long size);
void writeOutput(char *filename, int *a, long size);


// ./<executable_file> <input_file> <size> <output_file> <num_of_threads>
int main(int argc, char **argv)
{
	if(argc != 2){
        printf("Using: %s <size>\n", argv[0]);
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
	Count_sort(a,size);

	return 0;
}


void Count_sort(int *a, long n){
	long i, j, count;
	int *temp = malloc(n*sizeof(int));
	
	printf("Sorting...\n");
	
	double time_clocks = (double) clock();
	
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
	
	time_clocks = (double)clock() - time_clocks;

    double time = time_clocks/(double)CLOCKS_PER_SEC;
	
    printf("Sort completed\n\n");
	
	memcpy(a, temp, n*sizeof(int));
	free(temp);

    //print result
    for(i=0; i<n; i++){
        printf("%d ", a[i]);
    }
	
	printf("\n\nsequential countsort took %.5f seconds\n", time);
}

