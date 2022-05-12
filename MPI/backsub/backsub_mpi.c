#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>



int main(int argc, char* argv[]){
	int i,j,N;
	float **a, *b;
	float *local_x, *prev_x, *total_proc_x;
	int tag = 100;
	
	//MPI variables
	int rank, size;
	MPI_Status status;
	
	//time calculation
	double local_start, local_finish, local_elapsed, elapsed;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	
	
	if(argc != 2){
		if(rank == 0) printf("Using : %s <matrix_size>\n", argv[0]);
		return 0;
	}
	
	N = strtol(argv[1], NULL, 10);
	
	/* Allocate space for matrices */
	a = (float **) malloc ( N * sizeof ( float *) );
	for ( i = 0; i < N; i++) 
		a[i] = ( float * ) malloc ( N * sizeof ( float ) );
	b = ( float * ) malloc ( N * sizeof ( float ) );
	
	
	if(rank == 0){		
		srand ( time ( NULL));
		for (i = 0; i < N; i++) {
			b[i] = (float)rand()/(RAND_MAX*2.0-1.0);
			a[i][i] = 2.0+(float)rand()/(RAND_MAX*2.0-1.0);
			for (j = 0; j <= i; j++) 
				a[i][j] = (float)rand()/(RAND_MAX*2.0-1.0);
		} 
		
	}
	
	
	//broadcast data (a,b)
	if(rank == 0){
		int r;
		for(r=1; r<size; r++){
			for(i=0; i<N; i++){
				for(j=0; j<=i; j++){
					MPI_Send(&a[i][j],1,MPI_FLOAT,r,1,MPI_COMM_WORLD);
				}
				
			}
		}
	}
	else {
		for(i=0; i<N; i++){
			for(j=0; j<=i; j++){
				MPI_Recv(&a[i][j],1,MPI_FLOAT,0,1,MPI_COMM_WORLD,&status);
			}
			
		}
	}
	MPI_Bcast(b, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
	
	
	//initialize blocks
	int block_size = N/size;
	int *counts = (int *) malloc(size*sizeof(int));
	int *displacements = (int *) malloc(size*sizeof(int));
	for(i=0; i<size; i++){
		counts[i] = block_size;
		if(size-(i+1) < N%size) counts[i]++;
		if(i == 0) displacements[i] = 0;
		else displacements[i] = displacements[i-1] + counts[i-1];
	}
	
		
	local_x = (float *) malloc(counts[rank]*sizeof(float));
	for(i=0; i<counts[rank]; i++){
		local_x[i] = 0.0;
	}
	
	//all previous processes results (in the last process will contain the final result)
	prev_x = (float *) malloc(displacements[rank]*sizeof(float));
	
	if(rank == 0) printf("Size: %d\n", size);
	
	printf("Rank %d, Displacement: %d, Count: %d\n", rank, displacements[rank], counts[rank]);
	
	local_start = (double) MPI_Wtime();
	
	//calculation
	float sum;
	
	//if initialized as sequential problem
	if(size == 1){
		float *x = local_x;
		
		for(i=0; i<N; i++){
			sum = 0.0;
			for (j = 0; j < i; j++) {
				sum = sum + (x[j] * a[i][j]);
				//printf ("%d %d %f %f %f \t \n", i, j, x[j], a[i][j], sum);
			}	
			x[i] = (b[i] - sum) / a[i][i];
			//printf ("%d %f %f %f %f \n", i, b[i], sum, a[i][i], x[i]);
		}
		
		local_finish = (double) MPI_Wtime();
		
		elapsed = local_finish - local_start;
		
		/* Print result */
		if(N <= 20){
			for (i = 0; i < N; i++) {
				for (j = 0; j <= i; j++)
					printf ("%f \t", a[i][j]);	
				printf ("%f \t%f\n", total_proc_x[i], b[i]);
			}
		}
		
		
		/* Check result */
		if(N <= 20){
			for (i = 0; i < N; i++) {
				sum = 0.0;
				for (j = 0; j <= i; j++) 
					sum = sum + (total_proc_x[j]*a[i][j]);	
				if (fabsf(sum - b[i]) > 0.00001) {
					printf("%f != %f\n", sum, b[i]);
					printf("Validation Failed...\n");
				}
			}
		}
		
		printf("Backsub with %d process and problem size %d took %f seconds\n", size, N, elapsed);
		
		return 0;
	}
	
	
	/* Source of pipeline */
	if(rank == 0){
		for(i=0; i<counts[0]; i++){
			sum = 0.0;			
			for(j=0; j<i; j++){
				sum = sum + (local_x[j] * a[i][j]);
			}
			local_x[i] = (b[i] - sum) / a[i][i];
		}
		
		//send x to next process
		MPI_Send(local_x, counts[0], MPI_FLOAT, rank+1, tag, MPI_COMM_WORLD);
		printf("Process %d sent data to process %d\n", rank, rank+1);
		
		local_finish = (double) MPI_Wtime();
	}
	
	
	/* Main pipeline */   
	if(rank != 0 && rank != (size-1)){
		
		//receive all x from previous process
		MPI_Recv(prev_x, displacements[rank], MPI_FLOAT, rank-1, tag, MPI_COMM_WORLD, &status);
		printf("Process %d received data from process %d\n", rank, rank-1);
		
		for(i=displacements[rank]; i<(displacements[rank] + counts[rank]); i++){
			sum = 0.0;
			//unowned rows
			for(j=0; j<displacements[rank]; j++){
				sum = sum + (prev_x[j] * a[i][j]);
			}
			
			//owned rows
			for(j=displacements[rank]; j<i; j++){
				sum = sum + (local_x[j-displacements[rank]] * a[i][j]);
			}
			
			local_x[i-displacements[rank]] = (b[i] - sum) / a[i][i];
		}
		
		//concatenate prev and local x
		total_proc_x = (float *) malloc((displacements[rank] + counts[rank])*sizeof(float));
		for(i=0; i<displacements[rank]; i++){
			total_proc_x[i] = prev_x[i];
		}
		
		for(i=0; i<counts[rank]; i++){
			total_proc_x[i+displacements[rank]] = local_x[i];
		}
		
		//send x to next process
		MPI_Send(total_proc_x, displacements[rank+1], MPI_FLOAT, rank+1, tag, MPI_COMM_WORLD);
		
		local_finish = (double) MPI_Wtime();
	}
	
	
	/* Sink of pipeline */ 
	if(rank == (size-1)){
		//receive all x from previous process
		MPI_Recv(prev_x, displacements[rank], MPI_FLOAT, rank-1, tag, MPI_COMM_WORLD, &status);
		printf("Process %d received data from process %d\n", rank, rank-1);
		
		for(i=displacements[rank]; i<(displacements[rank] + counts[rank]); i++){
			sum = 0.0;
			//unowned rows
			for(j=0; j<displacements[rank]; j++){
				sum = sum + (prev_x[j] * a[i][j]);
			}
			
			//owned rows
			for(j=displacements[rank]; j<i; j++){
				sum = sum + (local_x[j-displacements[rank]] * a[i][j]);
			}
			
			local_x[i-displacements[rank]] = (b[i] - sum) / a[i][i];
		}
		
		//concatenate prev and local x
		float *total_proc_x = (float *) malloc((displacements[rank] + counts[rank])*sizeof(float));
		for(i=0; i<displacements[rank]; i++){
			total_proc_x[i] = prev_x[i];
		}
		
		for(i=0; i<counts[rank]; i++){
			total_proc_x[i+displacements[rank]] = local_x[i];
		}
		
		local_finish = (double) MPI_Wtime();

		/* Print result */
		if(N <= 20){
			for (i = 0; i < N; i++) {
				for (j = 0; j <= i; j++)
					printf ("%f \t", a[i][j]);	
				printf ("%f \t%f\n", total_proc_x[i], b[i]);
			}
		}
		
		
		/* Check result */
		if(N <= 20){
			for (i = 0; i < N; i++) {
				sum = 0.0;
				for (j = 0; j <= i; j++) 
					sum = sum + (total_proc_x[j]*a[i][j]);	
				if (fabsf(sum - b[i]) > 0.00001) {
					printf("%f != %f\n", sum, b[i]);
					printf("Validation Failed...\n");
				}
			}
		}
	}
	
	local_elapsed = local_finish - local_start;
	MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, size-1, MPI_COMM_WORLD);
	
	if(rank == (size-1)){
		printf("Backsub with %d processes and problem size %d took %f seconds\n", size, N, elapsed);
	}
	
	MPI_Finalize();
	
	return 0;
	
}

