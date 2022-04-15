#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>

void generate_list(int * X, int n);
void print_list(int * X, int n);
void merge(int * X, int n, int * tmp);
void mergesort_Parallel(int * X, int n, int * tmp, int threads);
void mergesort_Seq(int * X, int n, int * tmp);
bool isSorted(int * X, int size);


void main(int argc, char *argv[])
{
   int n, threads;
   int *data, *tmp;
  
   if (argc != 3) {
		printf ("Usage : %s <list size> <num_of_threads>\n", argv[0]);
      return;
   }
   n = strtol(argv[1], NULL, 10);
   threads = strtol(argv[2], NULL, 10);   
   data = (int *) malloc (sizeof(int)*n);
   tmp = (int *) malloc (sizeof(int)*n);
   

   printf("Generating List...\n");
   generate_list(data, n);
   printf("List Generated successfully\n");
   
   if(n<=20){
      printf("List Before Sorting...\n");
      print_list(data, n);
      printf("\n");
   }

   printf("Sorting...\n");
   
   double time = omp_get_wtime();
   
   //calculation
   omp_set_dynamic(0);
   omp_set_nested(1);
   mergesort_Parallel(data, n, tmp, threads);

   time = omp_get_wtime() - time;

   printf("Finished\n");

   if(isSorted(data, n)){
      printf("List sorted successfully\n");
   }
   else{
      printf("List is not sorted\n");
   }

   if(n <= 20){
      printf("\nList After Sorting...\n");
      print_list(data, n);
      printf("\n");   
   }

   printf("mergesort parallel sections with %d threads took %.5f seconds\n", threads, time);

}

void generate_list(int * X, int n) {
   int i;
   srand (time (NULL));
   for (i = 0; i < n; i++)
     X[i] = rand() % n; 
}

void print_list(int * X, int n) {
   int i;
   for (i = 0; i < n; i++) {
      printf("%d ",X[i]);
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


void mergesort_Parallel(int * X, int n, int * tmp, int threads){
   if(threads == 1){
      mergesort_Seq(X,n,tmp);
   }
   else if(threads > 1){
      //split the problem into <num_of_threads> parallel problems
      #pragma omp parallel sections
      {
         #pragma omp section
         {
            mergesort_Parallel(X, n/2, tmp, threads/2);
         }

         #pragma omp section
         {
            mergesort_Parallel(X+(n/2), n-(n/2), tmp+(n/2), threads-(threads/2));
         }

      }
      merge(X,n,tmp);
   }
}

void mergesort_Seq(int * X, int n, int * tmp)
{
   if (n < 2) return;

   mergesort_Seq(X, n/2, tmp); 
   mergesort_Seq(X+(n/2), n-(n/2), tmp);
   
   merge(X, n, tmp);
}


bool isSorted(int * X, int size){
   int i;
   for(i=0; i<size-1; i++){
      if(X[i] > X[i+1]) return false;
   }
   return true;
}