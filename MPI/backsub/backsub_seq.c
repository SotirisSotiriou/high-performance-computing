#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void main ( int argc, char *argv[] )  {

int   i, j, N;
float *x, *b, **a, sum;
char any;
double start, finish, elapsed;

	if (argc != 2) {
		printf ("Usage : %s <matrix size>\n", argv[0]);
                exit(1);
	}
	N = strtol(argv[1], NULL, 10);

	/* Allocate space for matrices */
	a = (float **) malloc ( N * sizeof ( float *) );
	for ( i = 0; i < N; i++) 
		a[i] = ( float * ) malloc ( (i+1) * sizeof ( float ) );
	b = ( float * ) malloc ( N * sizeof ( float ) );
	x = ( float * ) malloc ( N * sizeof ( float ) );

	/* Create floats between 0 and 1. Diagonal elents between 2 and 3. */
	srand ( time ( NULL));
	for (i = 0; i < N; i++) {
		x[i] = 0.0;
		b[i] = (float)rand()/(RAND_MAX*2.0-1.0);
		a[i][i] = 2.0+(float)rand()/(RAND_MAX*2.0-1.0);
		for (j = 0; j < i; j++) 
			a[i][j] = (float)rand()/(RAND_MAX*2.0-1.0);;
		for (j=i; j<N; j++)
			a[i][j] = 0.0;
	} 

	start = (double) clock()/CLOCKS_PER_SEC;
        /* Calulation */
	for (i = 0; i < N; i++) {
		sum = 0.0;
		for (j = 0; j < i; j++) {
			sum = sum + (x[j] * a[i][j]);
			//printf ("%d %d %f %f %f \t \n", i, j, x[j], a[i][j], sum);
		}	
		x[i] = (b[i] - sum) / a[i][i];
		//printf ("%d %f %f %f %f \n", i, b[i], sum, a[i][i], x[i]);
	}
	
	
	finish = (double) clock()/CLOCKS_PER_SEC;
	
	elapsed = finish - start;

    /* Print result */
    if(N <= 50)
		for (i = 0; i < N; i++) {
			for (j = 0; j <= i; j++)
				printf ("%f \t", a[i][j]);	
			printf ("%f \t%f\n", x[i], b[i]);
		}

   /* Check result */
   if(N <= 50)
		for (i = 0; i < N; i++) {
			sum = 0.0;
			for (j = 0; j <= i; j++) 
				sum = sum + (x[j]*a[i][j]);	
			if (fabsf(sum - b[i]) > 0.00001) {
				printf("%f != %f\n", sum, b[i]);
				printf("Validation Failed...\n");
			}
		}
	
	printf("Backsub with 1 process and problem size %d took %f seconds\n", N, elapsed);
}
