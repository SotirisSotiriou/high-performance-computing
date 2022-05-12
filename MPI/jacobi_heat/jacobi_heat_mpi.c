#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

#define maxn 12
#define iterations 100	
#define accuracy 50
#define start 100

void updateGhostPoints(double xlocal[][maxn], double ghostup[maxn], double ghostdown[maxn], int size, int rank);

int main(int argc, char *argv[]){
	int errcnt, value, toterr, i, j, k;
	double x[maxn][maxn];
	double localdiff, diff;
	//the local table with 1/4th of the normal size plus 2 for the ghost points
	double xlocal[(maxn/4)][maxn], xlocal2[(maxn/4)][maxn];
	double ghostup[maxn], ghostdown[maxn];
	char any;
	
	//MPI variables
	int size, rank;
	MPI_Status status;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	//initialize data
	for(i=0; i<maxn/size; i++){
		for(j=0; j<maxn; j++){
			xlocal[i][j] = 0;
			xlocal2[i][j] = 0;
		}
	}
	
	//initialize ghost points
	for(i=0; i<maxn; i++){
		ghostup[i] = -1;
		ghostdown[i] = -1;
	}
	
	if(rank == 1){
		xlocal[1][4] = start;
	}
	
	updateGhostPoints(xlocal, ghostup, ghostdown, size, rank);
	
	for(k=0; k<iterations; k++){
		
		//initialize local difference
		localdiff = 0.0;
		
		//create a heat source
		if(rank == 1){
			xlocal[1][4] = start;
		}
		
		
		if(rank == 0){
			for(i=1; i<maxn/size; i++){
				for(j=1; j<maxn-1; j++){
					if(i == maxn/size-1){
						xlocal2[i][j] = 0.25 * (xlocal[i-1][j] + ghostdown[j] + xlocal[i][j-1] + xlocal[i][j+1]);
						localdiff += (xlocal2[i][j] - xlocal[i][j]) * (xlocal2[i][j] - xlocal[i][j]);
					}
					else{
						xlocal2[i][j] = 0.25 * (xlocal[i-1][j] + xlocal[i+1][j] + xlocal[i][j-1] + xlocal[i][j+1]);
						localdiff += (xlocal2[i][j] - xlocal[i][j]) * (xlocal2[i][j] - xlocal[i][j]);
					}
				}
			}
		}
		
		if(rank != 0 && rank != size-1){
			for(i=0; i<maxn/size; i++){
				for(j=1; j<maxn-1; j++){
					if(i == 0){
						xlocal2[i][j] = 0.25 * (ghostup[j] + xlocal[i+1][j] + xlocal[i][j-1] + xlocal[i][j+1]);
						localdiff += (xlocal2[i][j] - xlocal[i][j]) * (xlocal2[i][j] - xlocal[i][j]);
					}
					else if(i == maxn/size-1){
						xlocal2[i][j] = 0.25 * (xlocal[i-1][j] + ghostdown[j] + xlocal[i][j-1] + xlocal[i][j+1]);
						localdiff += (xlocal2[i][j] - xlocal[i][j]) * (xlocal2[i][j] - xlocal[i][j]);
					}
					else{
						xlocal2[i][j] = 0.25 * (xlocal[i-1][j] + xlocal[i+1][j] + xlocal[i][j-1] + xlocal[i][j+1]);
						localdiff += (xlocal2[i][j] - xlocal[i][j]) * (xlocal2[i][j] - xlocal[i][j]);
					}
				}
			}
		}
		
		if(rank == size-1){
			for(i=0; i<maxn/size-1; i++){
				for(j=1; j<maxn-1; j++){
					if(i == 0){
						xlocal2[i][j] = 0.25 * (ghostup[j] + xlocal[i+1][j] + xlocal[i][j-1] + xlocal[i][j+1]);
						localdiff += (xlocal2[i][j] - xlocal[i][j]) * (xlocal2[i][j] - xlocal[i][j]);
					}
					else{
						xlocal2[i][j] = 0.25 * (xlocal[i-1][j] + xlocal[i+1][j] + xlocal[i][j-1] + xlocal[i][j+1]);
						localdiff += (xlocal2[i][j] - xlocal[i][j]) * (xlocal2[i][j] - xlocal[i][j]);
					}
				}
			}
		}
		
		
		//print local data
		int z;
		for(z=0; z<size; z++){
			if(z == rank){
				printf("Local data of rank %d in iteration %d:\n", rank, k);
				for(i=0; i<maxn; i++){
					printf("%5.0f ", ghostup[i]);
				}
				printf("\n");
				
				for(i=0; i<7*maxn; i++){
					printf("-");
				}
				printf("\n");
				
				for(i=0; i<maxn/size; i++){
					for(j=0; j<maxn; j++){
						printf("%5.0f ", xlocal[i][j]);
					}
					printf("\n");
				}
				
				for(i=0; i<7*maxn; i++){
					printf("-");
				}
				printf("\n");
				
				for(i=0; i<maxn; i++){
					printf("%5.0f ", ghostdown[i]);
				}
				printf("\n\n");
			}
			MPI_Barrier(MPI_COMM_WORLD);
		}
		
		
		MPI_Reduce(&localdiff, &diff, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		
		if(rank == 0){
			diff = sqrt(diff);
			printf("Iteration %d -> diff: %3.25f\n", k, diff);
		}
		
		MPI_Bcast(&diff, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		
		if(diff < accuracy){
			printf("\n\nConvergence in %d iterations\n\n", k);
			break;
		}
		
		//copy xlocal2 to xlocal
		for(i=0; i<maxn/size; i++){
			for(j=0; j<maxn; j++){
				xlocal[i][j] = xlocal2[i][j];
			}
		}
		
		updateGhostPoints(xlocal, ghostup, ghostdown, size, rank);
	}
	
	//put all data to x
	if(rank == 0){
		for(i=0; i<maxn/size; i++){
			for(j=0; j<maxn; j++){
				x[i][j] = xlocal[i][j];
			}
		}
		
		int r;
		for(r=1; r<size; r++){
			for(i=0; i<maxn/size; i++){
				for(j=0; j<maxn; j++){
					MPI_Recv(&x[r*(maxn/size)+i][j], 1, MPI_DOUBLE, r, 1, MPI_COMM_WORLD, &status);
				}
			}
		}
	}
	else{
		for(i=0; i<maxn/size; i++){
			for(j=0; j<maxn; j++){
				MPI_Send(&xlocal[i][j], 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
			}
		}
	}
	
	//print result
	if(rank == 0){
		printf("Global data:\n");
		for(i=0; i<maxn; i++){
			for(j=0; j<maxn; j++){
				printf("%5.0f ", x[i][j]);
			}
			printf("\n");
		}
		printf("\n\n");
	}
	
	MPI_Finalize();
	
	return 0;
}


void updateGhostPoints(double xlocal[][maxn], double ghostup[maxn], double ghostdown[maxn], int size, int rank){
	int errcnt, toterr, i, j;
	MPI_Status status;
	
	//send down unless i'm at the bottom, then receive from below
	//note the use of xlocal[i] for &xlocal[i][0]
	
	if(rank < size - 1){
		MPI_Send(xlocal[maxn/size-1], maxn, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
	}
	
	if(rank > 0){
		MPI_Recv(ghostup, maxn, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &status);
	}
	
	//send up unless i'm at the top
	if(rank > 0){
		MPI_Send(xlocal[0], maxn, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD);
	}
	
	
	if(rank < size - 1){
		MPI_Recv(ghostdown, maxn, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &status);
	}

}
