#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>
#include <omp.h>


int main (int argc, char *argv[]) {
	
	FILE *pFile;
	long file_size, match_size, pattern_size, total_matches;
	char * buffer;
	char * filename, *pattern;
	size_t result;
	int i, j /* , *match */;
	int threads;

	if (argc != 4) {
		printf ("Usage : %s <file_name> <string> <num_of_threads>\n", argv[0]);
		return 1;
	}
	filename = argv[1];
	pattern = argv[2];
	threads = strtol(argv[3], NULL, 10);
	
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
	
	/*
	match = (int *) malloc (sizeof(int)*match_size);
	if (match == NULL) {printf ("Malloc error\n"); return 5;}
	
	for (j = 0; j < match_size; j++){
		match[j]=0;
	}
	*/

	total_matches = 0;
        
    /* Brute Force string matching */
    double time = omp_get_wtime();

#pragma omp parallel for num_threads(threads) default(none) private(i,j) shared(match_size, pattern_size, pattern, buffer, total_matches/*, matches*/)
	for (j = 0; j < match_size; ++j) {
		for (i = 0; i < pattern_size && pattern[i] == buffer[i + j]; ++i);
		if (i >= pattern_size) {
			//match[j] = 1;
			#pragma omp critical
			{
				total_matches++;
			}
		}		
	}

	time = omp_get_wtime() - time;

	/*
	for (j = 0; j < match_size; j++){
		printf("%d", match[j]);
	}
	*/	
    printf("\nTotal matches = %ld\n", total_matches);
    printf("Total time for string matching seq: %.5f seconds\n", time );

	fclose (pFile);
	free (buffer);
	//free (match);

	return 0;
}
