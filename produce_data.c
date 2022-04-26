#include <stdio.h>
#include <stdlib.h>

#define N 128


//./<executable_file> <file> <data_size>
int main(int argc, char **argv){
	long size = strtol(argv[2], NULL, 10);
	FILE* f;
	
	if(argc != 3){
		printf("Using: <file> <data_size>\n");
		printf("Exit program...\n");
		exit(1);
	}
	
	f = fopen(argv[1], "w");
	if(f == NULL){
		printf("File Error\n");
		exit(1);
	}
	
	printf("Producing...\n");
	long i;
	for(i=0; i<size; i++){
		//random integer between 0 and N
		int number = rand()%N;
		fprintf(f, "%c", number);
	}
	printf("Finished\n");
	
	fclose(f);
	return 0;
}
