#include <stdio.h>
#include <math.h>
#include <time.h>

#define maxsize 50
#define iterations 100
#define row 20
#define col 20
#define start 100
#define accuracy 50

int main(int argc, char* argv[])
{
  int i, j, k;
  double table1[maxsize][maxsize], table2[maxsize][maxsize];
  double diff;

  /* initialize both tables*/

  for(i=0;i<maxsize;i++)
    for(j=0;j<maxsize;j++)
    {
      table1[i][j]=0;
      table2[i][j]=0;
    }

  double total_time_clocks = 0;

  /* repeate for each iteration */
  for(k = 0; k < iterations; k++) 
  {
    
    /* create a heat source */
    table1[row][col] = start;
    
    /* difference initialization */
    diff = 0.0;

    double iter_time_clocks = (double)clock();
    /* perform the calculations */
    for(i=1;i<maxsize-1;i++)
      for(j=1;j<maxsize-1;j++) {
        table2[i][j] = 0.25 *(table1[i-1][j] + table1[i+1][j] + table1[i][j-1] + table1[i][j+1]);
        diff += (table2[i][j]-table1[i][j])*(table2[i][j]-table1[i][j]);
      }
    iter_time_clocks = (double)clock()-iter_time_clocks;
    total_time_clocks += iter_time_clocks;
    

    // /* print result */
    // for(i=0;i<maxsize;i++)
    // {
    //   for(j=0;j<maxsize;j++)
    //     printf("%5.0f ",table2[i][j]);
    //   printf("\n");
    // }
    // printf("\n");
  
    /* print difference and check convergence */
    diff = sqrt(diff);
    printf("diff = %3.25f\n\n", diff);

    if (diff < accuracy) {
    printf ("\n\nConvergence in %d iterations\n\n", k);
          break;
    }	
  
  /* copy new table to old table */ 
    for(i=0;i<maxsize;i++)
      for(j=0;j<maxsize;j++)
        table1[i][j]=table2[i][j];
  }


  double time = total_time_clocks/(double)CLOCKS_PER_SEC;


  printf("Jacobi heat seq calculation took %.5f seconds\n", time);

  return 0;
}
