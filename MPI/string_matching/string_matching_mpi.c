#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>
#include <mpi.h>

int main(int argc, char* argv[]){
	FILE *pFile;
	long file_size, match_size, pattern_size, total_matches;
	char * buffer;
	char * filename, *pattern;
	size_t result;
	long i, j;
	
	char *local_buffer;
	long local_match_size, local_matches, local_buffer_size;
	
	//MPI variables
	int size, rank;
	MPI_Status status;
	int *counts, *displacements;
	
	//Time calculation
	double local_start, local_finish, local_elapsed, elapsed;
		
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	if (argc != 3) {
		if(rank == 0) printf ("Usage : %s <file_name> <string>\n", argv[0]);
		return 0;
	}
	
	if(rank == 0){
		filename = argv[1];
		pFile = fopen ( filename , "rb" );
		
		// obtain file size:
		fseek (pFile , 0 , SEEK_END);
		file_size = ftell (pFile);
		rewind (pFile);
		printf("file size is %ld\n", file_size);
		
		// allocate memory to contain the file:
		buffer = (char*) malloc (sizeof(char)*file_size);
		
		// copy the file into the buffer:
		result = fread(buffer,1,file_size,pFile);
		
		pattern = argv[2];
		pattern_size = strlen(pattern);
		
		match_size = file_size - pattern_size;
		
		counts = (int *) malloc(size*sizeof(int));
		displacements = (int *) malloc(size*sizeof(int));
		
		long start, end;
		for(i=0; i<size; i++){
			start = 0;
			for(j=0; j<i; j++){
				start += match_size/size;
				if(size-(j+1) < match_size%size) start++;
			}
			end = start + match_size/size + pattern_size - 1;
			counts[i] = end - start + 1;
			displacements[i] = start;
		}
		
		
		//for debugg
		printf("Counts: ");
		for(i=0; i<size; i++){
			printf("%d ", counts[i]);
		}
		printf("\n");
		
		printf("Displacements: ");
		for(i=0; i<size; i++){
			printf("%d ", displacements[i]);
		}
		printf("\n");
		
		
		local_buffer_size = counts[rank];
		local_buffer = (char *) malloc(local_buffer_size*sizeof(char));
		
		
		//send local sizes to processes
		for(i=1; i<size; i++){
			MPI_Send(&counts[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&pattern_size, 1, MPI_LONG, i, 0, MPI_COMM_WORLD);
			MPI_Send(pattern, pattern_size, MPI_CHAR, i, 0, MPI_COMM_WORLD);
		}
		
		MPI_Scatterv(buffer, counts, displacements, MPI_CHAR, local_buffer, local_buffer_size, MPI_CHAR, 0, MPI_COMM_WORLD);
		printf("Process %d sent data to slaves\n", rank);
				
	} else {
		MPI_Recv(&local_buffer_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&pattern_size, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD, &status);
		pattern = (char *) malloc(pattern_size*sizeof(char));
		MPI_Recv(pattern, pattern_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
		printf("Process %d received sizes\n", rank);
		
		local_buffer = (char *) malloc(local_buffer_size*sizeof(char));
		
		MPI_Scatterv(buffer, counts, displacements, MPI_CHAR, local_buffer, local_buffer_size, MPI_CHAR, 0, MPI_COMM_WORLD);
		printf("Process %d received data\n", rank);
	}
	
	printf("Rank: %d, Buffer Size: %ld\n", rank, local_buffer_size);
	
	local_start = (double) MPI_Wtime();
	
	//TODO: calculate matches
	local_matches = 0;
	for(j=0; j<local_buffer_size - pattern_size + 1; j++){
		for(i=0; i<pattern_size && pattern[i] == local_buffer[i + j]; ++i);
		if (i >= pattern_size) {
			local_matches++;
		}
	}
	
	printf("Process %d matches: %ld\n", rank, local_matches);
	
	local_finish = (double) MPI_Wtime();
	
	local_elapsed = local_finish - local_start;
	
	MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	
	MPI_Reduce(&local_matches, &total_matches, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	
	
	if(rank == 0){
		printf("Total matches = %ld\n", total_matches);
		printf("String matching with %d processes took %f seconds\n", size, elapsed);
		fclose(pFile);
	}
	
	free(local_buffer);
	
	MPI_Finalize();
	
	return 0;
}

