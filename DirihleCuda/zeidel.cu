#include<stdlib.h>
#include<math.h>
#include<iostream>
#include<time.h>

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

using namespace std;

__global__ void dirihle (float* u, float* f, float* v, int N, float h2, int rb, float* eps)
{
	
	__shared__ float s_u[BLOCK_DIM_X + 2][BLOCK_DIM_Y + 2];
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int tx = threadIdx.x+1;
	int ty = threadIdx.y+1;
	int k = j*N + i;
	if ( i < N && j < N )
	{
	    s_u[ty][tx] = u[k];
	    if ( ty == 1 && j > 0 )
		s_u[ty-1][tx] = u[k-N];
	    if ( ty == BLOCK_DIM_X && j < N-1 )
		s_u[ty+1][tx] = u[k+N];
	    if ( tx == 1 && i > 0 )
		s_u[ty][tx-1] = u[k-1];
	    if ( tx == BLOCK_DIM_Y && i < N-1 )
		s_u[ty][tx+1] = u[k+1];
	    if ( i == 0 )
		s_u[ty][tx-1] = u[k+N-2];
	    if( i == N-1 )
		s_u[ty][tx+1] = u[k-N+2];
	}
	__syncthreads();
	eps[0] = 0;
	if ( (i > 0 ) && ( i < N-1 ) && ( j > 0 ) && ( j < N-1 ) && ( k < N*N ) && ( i + j )%2 == rb ) {
		u[k] = 0.25*(s_u[ty-1][tx] + s_u[ty+1][tx] + s_u[ty][tx-1] + s_u[ty][tx+1] - h2*f[k]);
		}
		if ( eps[0] < abs(v[k] - u[k] )){ 
		eps[0] = abs(v[k] - u[k]);
	}
}
int main( int argc, char *  argv [] )
{
	int rows = 256;
	int count = 1;
	float* eps1;
	float* eps;
	float h =(float)1/(rows-1);
	int numBytes = sizeof(float)*rows*rows;
	float* u;
	float* f;
	float* v;
	u = ( float* )malloc( numBytes );
	f = ( float* )malloc( numBytes );
	v = ( float* )malloc( numBytes );
	eps = ((float*)malloc(sizeof(float)));
	eps1 = ((float*)malloc(sizeof(float)));
	eps[0] = 0; 
	for ( int i = 0; i < rows; i++ )
	    for ( int j = 0; j < rows; j++ ) {
		    float x = i*h;
		    float y = j*h;
		    f[i*rows + j] =4 + 2*x*x - 2*x + 2*y*y - 2*y;
		    u[i*rows + j] = 0;
		    v[i*rows + j] = (x*x - x + 1)*(y*y - y + 1);
	}
	for ( int i = 0; i < rows; i++ ) {
		float x = i*h;
		u[i*rows] = x*x - x + 1;
		u[i] = x*x - x + 1;
		u[i*rows+(rows-1)] = x*x - x + 1;
		u[(rows-1)*rows+i] = x*x - x + 1;
	}
	// allocate device memory
	float * devU = NULL;
	float * devV = NULL;
	float * devF = NULL;
	float * devE = NULL;
		
	cudaMalloc ( (void**)&devU, numBytes );
	cudaMalloc ( (void**)&devV, numBytes );
	cudaMalloc ( (void**)&devF, numBytes );
	cudaMalloc ((void**)&devE, sizeof(float));
	
	//set kernel launch configuration
	dim3 grid = dim3(16, 16);
	dim3 blocks  = dim3(16, 16);
	
	cudaMemcpy ( devU, u, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy ( devV, v, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy ( devF, f, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy ( devE, eps, sizeof(float), cudaMemcpyHostToDevice);
	clock_t t1, t2;
	t1 = clock();
        do{
            dirihle<<<grid, blocks>>>(devU, devF, devV, rows, h*h, 0, devE);
	    cudaMemcpy(eps1, devE, sizeof(float), cudaMemcpyDeviceToHost);
	    dirihle<<<grid, blocks>>>(devU, devF, devV, rows, h*h, 1, devE);
	    cudaMemcpy(eps, devE, sizeof(float), cudaMemcpyDeviceToHost);
	    cerr<<count<<" "<<eps[0]<<" "<<eps1[0]<<endl;
	    count++;
	    }
	    while (count < 35300 || eps[0] > 0.005 || eps1[0] > 0.005);
	    cudaMemcpy(u,devU,numBytes,cudaMemcpyDeviceToHost);
	t2 = clock();
	cerr<<" "<<((float)(t2 - t1))/(CLOCKS_PER_SEC)<<" sec" <<endl;
	for (int i = 0; i < rows; i++ )
	    for ( int j = 0; j < rows; j++ ) {
	        cout << i*h << " " << j*h << " " << u[i*rows+j] <<endl;
	}
	delete [] u;
	delete [] f;
	delete [] v;
	delete [] eps;
	
	cudaFree ( devU );
	cudaFree ( devV );
	cudaFree ( devF );
	cudaFree ( devE );
	
	return 0;
}




