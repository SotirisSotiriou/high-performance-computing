
//compile command: mpicc -o mergesort_mpi mergesort_mpi -lm

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <mpi.h>
#include <math.h>
#include <string.h>

bool isPowerOfTwo(int n);
void generate_list(int * x, int n);
void print_list(int * x, int n);
bool is_sorted(int *X, int n);
void merge(int * X, int n, int * tmp);
void mergesort(int * X, int n, int * tmp);
void merge_mpi(int * half1, int * half2, int * result, int n);
int *mergesort_mpi(int *local_data, int height, int rank, int local_n, MPI_Comm comm, int comm_sz, int *data, int n);


int main(int argc, char *argv[]){
	//world variables
	int *data, n, height;
	int i,j;
	
	//local variables
	int *local_data, local_n;
	
	//MPI variables
	int rank, size;
	int *counts, *displacements;
	
	//time calculation
	double local_start, local_finish, local_elapsed, elapsed;
	double zeroProcStart, zeroProcFinish, zeroProcElapsed;
	double procStart, procFinish, procElapsed;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	//check if args are 2
	if(argc != 2){
		if(rank == 0) printf("Using : %s <array_size>\n", argv[0]);
		return 0;
	}
	
	
	//check if odd processes
	if(!isPowerOfTwo(size)){
		if(rank == 0) printf("Number of processes must be power of 2\n");
		return 0;
	}
	
	//read problem size
	n = strtol(argv[1], NULL, 10);
	
	//calculate total height of the tree
	height = log2(size);
	
	//if process 0, allocate memory for global array and initialize data
	if(rank == 0){
		data = (int *) malloc(n*sizeof(int));
		generate_list(data,n);	
		if(n<=50) print_list(data,n);
	}
	
	
	//allocate memory for local data, scatter to fill with values
	counts = (int *) malloc(size*sizeof(int));
	displacements = (int *) malloc(size*sizeof(int));
	int start, end;
	for(i=0; i<size; i++){
		start = 0;
		for(j=0; j<i; j++){
			start += n/size;
			if(size-(j+1) < n%size) start++;
		}
		end = start + n/size;
		if(size - (i+1) < n%size) end++;
		counts[i] = end - start;
		displacements[i] = start;
	}
	
	local_n = counts[rank];
	local_data = (int *) malloc(local_n*sizeof(int));
	
	if(rank == 0) MPI_Scatterv(data, counts, displacements, MPI_INT, local_data, local_n, MPI_INT, 0, MPI_COMM_WORLD);
	else MPI_Scatterv(NULL, NULL, NULL, MPI_INT, local_data, local_n, MPI_INT, 0, MPI_COMM_WORLD);
	
	//for debugging
	if(rank == 0){
		printf("Counts: ");
		print_list(counts,size);
		printf("Displacements: ");
		print_list(displacements,size);
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	if(n<=50) {
		print_list(local_data,local_n);
	}
	
	//start timing
	local_start = (double) MPI_Wtime();
	
	//merge sort
	if(rank == 0){
		zeroProcStart = (double) MPI_Wtime();
		
		data = mergesort_mpi(local_data, height, rank, local_n, MPI_COMM_WORLD, size, data, n);
		
		zeroProcFinish = (double) MPI_Wtime();
		zeroProcElapsed = zeroProcFinish - zeroProcStart;
		printf("Process 0 took %f seconds\n", zeroProcElapsed);
	} else {
		procStart = (double) MPI_Wtime();
		
		mergesort_mpi(local_data, height, rank, local_n, MPI_COMM_WORLD, size, NULL, n);
	
		procFinish = (double) MPI_Wtime();
		procElapsed = procFinish - procStart;
		printf("Process %d took %f seconds\n", rank, procElapsed);
	}
	
	//end timing
	local_finish = (double) MPI_Wtime();
	local_elapsed = local_finish - local_start;
	MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	
	
	//vallidate result
	if(rank == 0){
		if(n <= 50) print_list(data,n);
		printf("The result is ");
		is_sorted(data,n) ? printf("correct\n") : printf("not correct\n");
		printf("Merge sort with %d processes took %f seconds\n", size, elapsed);
		free(data);
	}
		
	MPI_Finalize();
	
	
	return 0;
}


bool isPowerOfTwo(int n){
	return ( ceil(log2(n)) == floor(log2(n)) );
}


void generate_list(int * x, int n) {
   int i;
   srand (time (NULL));
   for (i = 0; i < n; i++)
     x[i] = rand() % n; 
}


void print_list(int * x, int n) {
   int i;
   for (i = 0; i < n; i++) {
      printf("%d ",x[i]);
   } 
   printf("\n");
}


bool is_sorted(int *X, int n){
	int i;
	
	for(i=1; i<n; i++){
		if(X[i] < X[i-1]) return false;
	}
	return true;
}


void merge(int * X, int n, int * tmp) {
   int i = 0;
   int j = n/2;
   int ti = 0;

   while (i<n/2 && j<n) { /* merge */
      if (X[i] < X[j]) {
         tmp[ti] = X[i];
         ti++; i++;
      } else {
         tmp[ti] = X[j];
         ti++; j++;
      }
   }
   while (i<n/2) { /* finish up lower half */
      tmp[ti] = X[i];
      ti++; i++;
   }
   while (j<n) { /* finish up upper half */
       tmp[ti] = X[j];
       ti++; j++;
   }
   memcpy(X, tmp, n*sizeof(int));

} 


void mergesort(int * X, int n, int * tmp)
{
   if (n < 2) return;

   mergesort(X, n/2, tmp); 
   mergesort(X+(n/2), n-(n/2), tmp);
   
   merge(X, n, tmp);
}


void merge_mpi(int * half1, int * half2, int * result, int n){
	int i=0;
	int j=0;
	int ti=0;
	int *tmp = (int *) malloc(n*2*sizeof(int));
	
	while(i<n && j<n){
		if(half1[i] < half2[j]){
			tmp[ti] = half1[i];
			ti++; i++;
		} else {
			tmp[ti] = half2[j];
			ti++; j++;
		}
	}
	
	while(i<n){
		tmp[ti] = half1[i];
		ti++; i++;
	}
	
	while(j<n){
		tmp[ti] = half2[j];
		ti++; j++;
	}
	
	memcpy(result, tmp, 2*n*sizeof(int));
}


int *mergesort_mpi(int *local_data, int height, int rank, int local_n, MPI_Comm comm, int comm_sz, int *data, int n){
	int parent, rightChild, myHeight;
	int *half1, *half2, *mergeResult;
	int *tmp;
	MPI_Status status;
	int i;
	
	myHeight = 0;
	tmp = (int *) malloc(local_n*sizeof(int));
	mergesort(local_data,local_n,tmp);
	half1 = local_data;
	free(local_data);
	
	parent = 0;
	while(myHeight < height){ //not yet at top
		if(rank != 0) parent = (rank & (~(1 << myHeight)));
		
		if(parent == rank){ //left child
			rightChild = (rank | (1 << myHeight));
			
			// allocate memory and receive array of right child
			half2 = (int*) malloc (local_n * sizeof(int));			
			MPI_Recv(half2, local_n, MPI_INT, rightChild, 0, MPI_COMM_WORLD, &status);
			
			// allocate memory for result of merge
  		    mergeResult = (int*) malloc (local_n * 2 * sizeof(int));
  		    
  		    // merge half1 and half2 into mergeResult
  		    for(i=0; i<local_n; i++){
				mergeResult[i] = half1[i];
				mergeResult[i+local_n] = half2[i];
			}
			
			merge_mpi(half1, half2, mergeResult, local_n);
  		    
  		    
  		    // reassign half1 to merge result
            half1 = mergeResult;
			local_n = local_n * 2;  // double size
			
			free(half2); 
			mergeResult = NULL;

            myHeight++;
		
		} else { //right child
			//send local data to parent
			MPI_Send(half1, local_n, MPI_INT, parent, 0, MPI_COMM_WORLD);
			if(myHeight != 0) free(half1);  
			myHeight = height;
		}
	}
	
	
	
	free(tmp);
	
	if(rank == 0) {
		data = half1; //reassign full merged array to data
	}
	
	return data;
}
