#include <stdio.h> 
#include <stdlib.h> 
#include <omp.h>

#define N 128
#define base 0


int main (int argc, char *argv[]) {
	
	FILE *pFile;
	FILE *output;
	long file_size;
	char * buffer;
	char * filename;
	char * output_filename;
	size_t result;
	int i, j, freq[N];
	int threads;

	if (argc != 3) {
		printf ("Usage : %s <input_file_name> <num_of_threads>\n", argv[0]);
		return 1;
	}
	filename = argv[1];
	pFile = fopen ( filename , "rb" );
	if (pFile==NULL) {printf ("File error\n"); return 2;}
	
	threads = strtol(argv[2], NULL, 10);

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
	
	for (j=0; j<N; j++){
		freq[j]=0;
	}

	//calculate character frequenccy
	omp_set_num_threads(threads);
	int *local_freq[omp_get_num_threads()];

	double time = omp_get_wtime();

	#pragma omp parallel private(i) shared(file_size, buffer, freq, local_freq)
	{
		int id = omp_get_thread_num();

		local_freq[id] = malloc(N*sizeof(int));

		#pragma omp for
		for(i=0; i<file_size; i++){
			local_freq[id][buffer[i] - base]++;
		}

		for(j=0; j<N; j++){
			#pragma omp critical
			{
				freq[j] += local_freq[id][j];
			}
		}
	}		
	
	time = omp_get_wtime() - time;
	
	fclose (pFile);
	
	//print result
    for(i=0; i<N; i++){
        printf("%d = %d\n", i, freq[i]);
    }

	printf("\nchar_freq_parallel with %d threads took %.2f seconds\n", threads, time);
	
	fclose(output);
	free (buffer);

	return 0;
}

