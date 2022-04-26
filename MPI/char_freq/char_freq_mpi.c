#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 128
#define base 0

int main(int argc, char* argv[]){
	FILE *pFile;
	long file_size;
	char * buffer;
	char * filename;
	size_t result;
	int i, j, freq[N];
	
	int local_freq[N], local_n;
	char *local_buffer;
	int min_block_size;
	
	//MPI variables
	int size, rank;
	MPI_Status status;
	
	//Time calculation
	double local_start, local_finish, local_elapsed, elapsed;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	if(argc != 2){
		if(rank == 0) printf("Usinng : %s <file_name>\n", argv[0]);
		return 0;
	}

	
	
	if(rank == 0){
		filename = argv[1];
		pFile = fopen ( filename , "rb" );
		
		//obtain file size
		fseek (pFile , 0 , SEEK_END);
		file_size = ftell (pFile);
		rewind (pFile);
		printf("file size is %ld\n", file_size);
		
		// allocate memory to contain the file:
		buffer = (char*) malloc (sizeof(char)*file_size);
		if(buffer == NULL) printf("Memory error...\n");
		
		result = fread(buffer,1,file_size,pFile);
		if(result != file_size) printf("Reading error...\n");
		
		//Initialize freqs
		for(i=0; i<N; i++){
			freq[i] = 0;
		}
		
		//Send file size to all processes
		for(i=0; i<size; i++){
			MPI_Send(&file_size, 1, MPI_LONG, i, 0, MPI_COMM_WORLD);
		}
		
	} else {
		MPI_Recv(&file_size, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD, &status);
	}
	
	
	
	min_block_size = file_size/size;
	local_n = min_block_size;
	if(size-(rank+1) < file_size%size) local_n += 1;
	
	
	//Devide buffer to processes
	local_buffer = (char *) malloc(local_n*sizeof(char));
	MPI_Scatter(buffer, local_n, MPI_CHAR, local_buffer, local_n, MPI_CHAR, 0, MPI_COMM_WORLD);
	
	if(rank == 0) free(buffer);
	
	printf("Rank: %d, Buffer Size: %d, Problem Size: %ld\n", rank, local_n, file_size);
	
	//Initialize local freqs
	for(i=0; i<N; i++){
		local_freq[i] = 0;
	}
	
	local_start = (double) MPI_Wtime();
	
	//Find local freqs
	for(i=0; i<local_n; i++) {
		local_freq[local_buffer[i] + base]++;
	}
	
	
	local_finish = (double) MPI_Wtime();
	
	local_elapsed = local_finish - local_start;
	
	MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	
	//Sum all freqs to process 0
	for(i=0; i<N; i++){
		MPI_Reduce(&local_freq[i], &freq[i], 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	}
	
	//Print result in process 0
	if(rank == 0){
		for(i=0; i<N; i++){
			printf("%d = %d\n", i+base, freq[i]);
		}
		
		printf("Char freq with %d processes took %f seconds\n", size, elapsed);
		
		fclose(pFile);
	}
	
	
	MPI_Finalize();
	
	return 0;
}
