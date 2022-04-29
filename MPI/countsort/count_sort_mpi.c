#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <mpi.h>

#define UPPER 1000
#define LOWER 0


int main(int argc, char* argv[]){
	int *x, *y, n, *places;	
	int local_n, *local_places;
	int *counts, *displacements;
	int i, j;
	int comm_sz, my_rank;
	double elapsed, local_elapsed, local_start, local_finish;
	MPI_Status status;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	
	if(argc != 2){
		if(my_rank == 0) printf("Using : %s <array_size>\n", argv[0]);
		return 0;
	}

	n = strtol(argv[1], NULL, 10);
	x = malloc(n*sizeof(int));
	int min_block_size = n/comm_sz;
	local_n = min_block_size;
	if(comm_sz-(my_rank+1) < n%comm_sz) local_n += 1;
	
	printf("Rank: %d, Size: %d, Problem Size: %d\n", my_rank, local_n, n);
	
	if(my_rank == 0){
		//initialize data
		for(i=0; i<n; i++){
			x[i] = n - i;
			//x[i] = (rand() % (UPPER - LOWER + 1)) + LOWER;	
		}
		
		places = (int *) malloc(n*sizeof(int));
		
		//send data to all processes
		if(comm_sz > 1){
			for(j=1; j<comm_sz; j++){
				for(i=0; i<n; i++){
					MPI_Send(&x[i], 1, MPI_INT, j, 0, MPI_COMM_WORLD);
				}
				//printf("Process 0 sent data to process %d\n", j);			
			}
		}
	
	}
	else{
		//recv data from process 0		
		for(i=0; i<n; i++){
			MPI_Recv(&x[i], 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		}
		//printf("Process %d received data from process %d\n", my_rank, 0);
		
	}
	
	//Initialize counts and displacements
	counts = (int *) malloc(comm_sz*sizeof(int));
	displacements = (int *) malloc(comm_sz*sizeof(int));
	for(i=0; i<comm_sz; i++){
		counts[i] = min_block_size;
		if(comm_sz-(i+1) < n%comm_sz) counts[i]++;
		int rank_start=0;
		for(j=0; j<i; j++){
			rank_start += counts[j];
		}
		displacements[i] = rank_start;
	}
	
	local_places = (int *) malloc(counts[my_rank]*sizeof(int));
		
	local_start = (double) MPI_Wtime();
	
	//calculate local places
	int count=0;
	for(j=displacements[my_rank]; j<displacements[my_rank]+counts[my_rank]; j++){
		int my_num = x[j];
		int my_place = 0;
		for(i=0; i<n; i++){
			if ((my_num > x[i]) || ((my_num == x[i]) && (j < i))) my_place++;
		}
		local_places[count] = my_place;
		count++;
	}
	
	local_finish = (double) MPI_Wtime();
	
	local_elapsed = local_finish - local_start;
	
	MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	
	if(my_rank == 0) MPI_Gatherv(local_places, local_n, MPI_INT, places, counts, displacements, MPI_INT, 0, MPI_COMM_WORLD);
	else MPI_Gatherv(local_places, local_n, MPI_INT, NULL, NULL, NULL, MPI_INT, 0, MPI_COMM_WORLD);
	
	
	if(my_rank == 0){
		y = malloc(n*sizeof(int));
		for(i=0; i<n; i++){
			y[places[i]] = x[i];
		}	
		
		//validate result
		bool correct = true;
		for(i=1; i<n; i++){
			if(y[i] < y[i-1]){
				correct = false;
				break;
			}
		}
		printf("The sorting result is ");
		if(correct) printf("correct\n");
		else printf("not correct\n");
		
		printf("Count sort with %d processes took %f seconds\n", comm_sz, elapsed);
	}
	
	MPI_Finalize();
	
	
	return 0;
}
