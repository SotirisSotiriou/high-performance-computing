#include <stdio.h>
#include <stdlib.h>

#define N 128

int main(void){
	long size;
	int i;
	long j;
	char filename[50];
	
	
	size = 1000000;
	for(i=0; i<4; i++){
		sprintf(filename, "input_size_%ld", size);
		
		FILE *f;
		f = fopen(filename, "w");
		if(f == NULL){
			printf("File Error\n");
			exit(1);
		}
		
		printf("Producing file with size %ld\n", size);
		
		for(j=0; j<size; j++){
			int number = rand()%N;
			fprintf(f, "%c", number);
		}
		fclose(f);
		
		printf("Fnished\n\n");
		
		size *= 10;
	}
	
	return 0;
}
