#include "mpi.h"
#include <stdio.h>

int main (int argc, char *argv[])
{
	int n = 100000000, myid, numprocs, i;
	double mypi, pi, h, sum, x;
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	h = 1.0 / (double) n;
	x = 0;
	sum = 0.0;
	double t0, t1;
	int blocksize = n / numprocs;
	t0 = MPI_Wtime();
	for (i = myid * blocksize; i < blocksize * (myid + 1); i++)
	{
		x = i * h;
		sum += 4.0/(1.0 + x * x);
	}
	sum *= h;
	MPI_Reduce(&sum, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	t1 = MPI_Wtime();
	if (myid == 0) printf("pi is approximately %.16f\nTotal time spent: %f\n", pi, t1 - t0);
	MPI_Finalize();
	return 0;
}
