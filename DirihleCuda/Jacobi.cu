#include<stdlib.h>
#include<math.h>
#include<iostream>
#include<time.h>

#define N 512
#define BLOCKS 64 

using namespace std;

__global__ void Jacobi(double* u1, double* u2, double* f, double* ut, double h2, double* dmax)
{
	dmax[0] = 0; // max error	
	double dm = 0; // temporary value of error
	
	int i = blockIdx.x*blockDim.x + threadIdx.x; 
	int j = blockIdx.y*blockDim.y + threadIdx.y; 
	int k = j*N + i;		 

	if((i > 0)&&(i < N-1)&&(j > 0)&&(j < N-1)&&(k < N*N))
	{
		u2[k] = 0.25*(u1[k-1] + u1[k+1] + u1[k-N] + u1[k+N] - h2*f[k]);
		dm = fabs(ut[k]-u2[k]);
		if (dmax[0] < dm) dmax[0] = dm;
		
		u1[k] = u2[k];
	}
}

void boundaryConditions(double * U, double h)
{
	for (int i = 0; i < N; i++) {
		double x = i*h;
		U[i*N] = x*x - x + 1;
		U[i] = x*x - x + 1;
		U[i*N+N-1] = x*x - x + 1;
		U[(N-1)*N+i] = x*x - x + 1;
	}
}

int main( int argc, char *  argv [] )
{
	//double EPS = 0.0005;
	double h =(double)1/(double)N;
	int numBytes = sizeof(double)*N*N;
	double* U1;
	double* U2;
	double* f;
	double* UT;
	double* dmax;
	
	U1 = (double*)malloc(numBytes);
	U2 = (double*)malloc(numBytes);
	f = (double*)malloc(numBytes);
	UT = (double*)malloc(numBytes);
	dmax = (double*)malloc(sizeof(double));
	dmax[0] = 1;

	/* Matrix Initialization */
	for(int i = 0; i < N; i++)
	{
		double x = i*h;
		for(int j = 0; j < N; j++)
		{
		    double y = j*h;
		    f[i*N+j] = 4 + 2*x*x - 2*x + 2*y*y - 2*y; // rightSide 1st variant
		    U1[i*N+j] = 0;
		    U2[i*N+j] = 0;
		}
	}

	boundaryConditions(U1,h);
	boundaryConditions(U2,h);
	
	/* theoretical Solution */ 
	for (int i = 0; i < N; i++)
	{
		double x = i*h;
		for (int j = 0; j < N; j++)
		{
			double y = j*h;
			UT[i*N+j] = (x*x - x + 1)*(y*y - y + 1);
		}
	}
	
			// allocate device memory
	double * adev = NULL;
	double * bdev = NULL;
	double * cdev = NULL;
	double * tdev = NULL;
	double * devD = NULL;
	
	cudaMalloc ( (void**)&adev, numBytes );
    cudaMalloc ( (void**)&bdev, numBytes );
    cudaMalloc ( (void**)&cdev, numBytes );
    cudaMalloc ( (void**)&tdev, numBytes );
	cudaMalloc ( (void**)&devD, sizeof(double) );

	cudaMemcpy ( adev, U1, numBytes, cudaMemcpyHostToDevice );
    cudaMemcpy ( bdev, U2, numBytes, cudaMemcpyHostToDevice );
    cudaMemcpy ( cdev, f, numBytes, cudaMemcpyHostToDevice );
    cudaMemcpy ( tdev, UT, numBytes, cudaMemcpyHostToDevice );
	cudaMemcpy ( devD, dmax, sizeof(double), cudaMemcpyHostToDevice ); 
	
			// set kernel launch configuration
	dim3 threads = dim3(32, 32, 1);
	dim3 blocks  = dim3(16, 16, 1);
	
            // create cuda event handles
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;

    cudaEventCreate ( &start );
    cudaEventCreate ( &stop );

            // asynchronously issue work to the GPU (all to stream 0)
    cudaEventRecord ( start, 0 );	

	int k = 0; // iteration counter

	for (int i = 0; i < 59000 && dmax[0] > 0.00001; i++)
	{
		Jacobi<<<threads, blocks,0>>>(adev, bdev, cdev, tdev, h*h, devD);	
		cudaMemcpy ( dmax, devD, sizeof(double), cudaMemcpyDeviceToHost);
		if(k%100 == 0) 
			cerr << k <<" "<< dmax[0] << "\n";

		k++;
	}


	
	cudaMemcpy ( U2, bdev, numBytes, cudaMemcpyDeviceToHost );
        
	cudaEventRecord ( stop, 0 );

    cudaEventSynchronize ( stop );
    cudaEventElapsedTime ( &gpuTime, start, stop );
	
	cerr << "Executing time: " << (float)gpuTime << " milliseconds" << "\n";
	cerr << "Total number of iterations: "  << k << "\n";
	cerr << "Error: " << dmax[0] << "\n";
	
	for(int i = 0; i < N ; i++)
	{
		for(int j = 0; j < N ; j++)
		{
			cout << i*h << " " << j*h << " " << U2[i*N+j] << "\n";
		}	
	}
	
				// release resources
    cudaEventDestroy ( start );
    cudaEventDestroy ( stop  );	
	
    cudaFree         ( adev  );
    cudaFree         ( bdev  );
    cudaFree         ( cdev  );
    cudaFree         ( tdev  );
	cudaFree		 ( devD  );
	
	free(U1);
	free(U2);
	free(UT);
	free(f);
	free(dmax);
	
	
	return 0;
}

