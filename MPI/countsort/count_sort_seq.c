#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

#define UPPER 1000
#define LOWER 0

int main(int argc, char *argv[])
{
   int *x, *y;
   int i, j, my_num, my_place;
   double t_total;
   
   if (argc != 2) {
		printf ("Usage : %s <array_size>\n", argv[0]);
		return 1;
   }
   
   
   int n = strtol(argv[1], NULL, 10);
   x = ( int * ) malloc ( n * sizeof ( int ) );
   y = ( int * ) malloc ( n * sizeof ( int ) );

   for (i=0; i<n; i++)
		x[i] = n - i;
		//x[i] = (rand() % (UPPER - LOWER + 1)) + LOWER;
	
	double t_total_clocks = (double) clock();
		
   for (j=0; j<n; j++) {

     my_num = x[j];
     my_place = 0;
     for (i=0; i<n; i++)
		if ((my_num > x[i]) || ((my_num == x[i]) && (j < i))) 
			my_place++;
     y[my_place] = my_num;
   }  
   
   t_total_clocks = (double) clock() - t_total_clocks;
   t_total = t_total_clocks/(double)CLOCKS_PER_SEC;
   
   
   //print result
   /*
   for (i=0; i<n; i++) 
		printf("%d ", y[i]);
	*/
	
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
	
	printf("Count sort seq took %f seconds\n", t_total);
			
   return 0;
}
