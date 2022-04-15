#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

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

bool isSorted(int *X, int size){
   int i;
   for(i=0; i<size-1; i++){
      if(X[i] > X[i+1]) return false;
   }
   return true;
}


void main(int argc, char *argv[])
{
   int n;
   int *data, *tmp;
  
   if (argc != 2) {
		printf ("Usage : %s <list size>\n", argv[0]);
   }
   n = strtol(argv[1], NULL, 10);
   data = (int *) malloc (sizeof(int)*n);
   tmp = (int *) malloc (sizeof(int)*n);
   
   printf("Generating List...\n");
   generate_list(data, n);
   printf("List Generated successfully\n");
   // printf("List Before Sorting...\n");
   // print_list(data, n);
   
   double time_clocks = (double) clock();

   printf("Sorting...\n");

   //calculation
   mergesort(data, n, tmp);
   
   time_clocks = (double) clock() - time_clocks;
   double time = time_clocks/(double)CLOCKS_PER_SEC;

   printf("Finished\n");

   if(isSorted(data, n)){
      printf("List sorted successfully\n");
   }
   else{
      printf("List is not sorted\n");
   }
   
   // printf("\nList After Sorting...\n");
   // print_list(data, n);
   // printf("\n");

   printf("mergesort seq took %.5f seconds\n", time);

}

