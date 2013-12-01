#include<stdlib.h>
#include<math.h>
#include<iostream>
#include<time.h>

#define omega 1.5

using namespace std;

__global__ void calculateU(double* u, double* f, double* pu, int N, double h2, int rb, double * e)
{
	__shared__ double s_u[10][10];
	e[0]=0;
	int i = blockIdx.x*blockDim.x + threadIdx.x; //итератор по "большой" матрице
	int j = blockIdx.y*blockDim.y + threadIdx.y; //итератор по "большой" матрице
	int tx = threadIdx.x+1;
	int ty = threadIdx.y+1;
	int k = j*N + i;		 //итератор по "большой" матрице
	if(i<N && j<N)
	{
	    s_u[ty][tx] = u[k];
	    if(ty==1 && j>0)
		s_u[ty-1][tx] = u[k-N];
	    if(ty==8 && j<N-1)
		s_u[ty+1][tx] = u[k+N];
	    if(tx==1 && i>0)
		s_u[ty][tx-1] = u[k-1];
	    if(tx==8 && i<N-1)
		s_u[ty][tx+1] = u[k+1];
		
	    if(i==0)
		s_u[ty][tx-1] = u[k+N-2];
	    if(i==N-1)
		s_u[ty][tx+1] = u[k-N+2];
	}
	__syncthreads();
	if((i>0)&&(i<N-1)&&(j>0)&&(j<N-1)&&(k<N*N)&&(i+j)%2==rb){
	    u[k] = omega*0.25*(h2*f[k] +s_u[ty-1][tx] + s_u[ty+1][tx] + s_u[ty][tx-1] + s_u[ty][tx+1] - 4*s_u[ty][tx])+ s_u[ty][tx];
	    if(e[0]<abs(pu[k]-u[k]))
		e[0] = abs(pu[k]-u[k]);
	}
}

int main( int argc, char *  argv [] )
{
	double h =(double)1/(double)1024;
	double h2=h*h;
	int numBytes = sizeof(double)*1024*1024;
	double* U;
	double* f;
	double* pU;
	double* eps;
	U = (double*)malloc(numBytes);
	f = (double*)malloc(numBytes);
	pU = (double*)malloc(numBytes);
	eps = (double*)malloc(sizeof(double));
	eps[0]=1;
	//инициализация матриц и точного решения

	for(int i=0; i<1024; i++)
	{
		double x = i*h;
		for(int j=0; j<1024; j++)
		{
		    double y = j*h;
		    f[i*1024+j] = 4;
		    U[i*1024+j] = 0;
		    pU[i*1024+j] = -x*x -y*y +x +y;
		}

	}
	//начальные условия
	for(int i=0; i<1024 ; i++)
	{
		double x = i*h;
		U[i*1024] = - x*x + x;
		U[i] = - x*x + x;
		U[i*1024+1023] = - x*x + x;
		U[1023*1024+i] = - x*x + x;
		pU[i*1024] = - x*x + x;
		pU[i] = - x*x + x;
		pU[i*1024+1023] = - x*x + x;
		pU[1024*1023+i] = - x*x + x;
	}
	// allocate device memory
	double * devU = NULL;
	double * devpU = NULL;
	double * devF = NULL;
	double * devE = NULL;
	
	cudaMalloc ( (void**)&devU, numBytes );
	cudaMalloc ( (void**)&devpU, numBytes );
	cudaMalloc ( (void**)&devF, numBytes );
	cudaMalloc ( (void**)&devE, sizeof(double));
	
	// set kernel launch configuration
	dim3 threads = dim3(128, 128,1);
	dim3 blocks  = dim3(8, 8,1);
	
	cudaMemcpy(devU,U,numBytes,cudaMemcpyHostToDevice);
	cudaMemcpy(devpU,pU,numBytes,cudaMemcpyHostToDevice);
	cudaMemcpy(devF,f,numBytes,cudaMemcpyHostToDevice);
	cudaMemcpy(devE,eps,sizeof(double),cudaMemcpyHostToDevice);
	//запуск ядерной функции
	int count = 0;
	clock();
	for(int i = 0; (i<50000)&&(eps[0]>0.0005); i++){
    	    //eps[0]=0;
    	    calculateU<<<threads,blocks,0>>>(devU,devF,devpU,1024,h2,0,devE);
    	    calculateU<<<threads,blocks,0>>>(devU,devF,devpU,1024,h2,1,devE);
    	    cudaMemcpy(eps,devE,sizeof(double),cudaMemcpyDeviceToHost);
    	    count = i;
	}
	//вывод результатов
	cudaMemcpy(U,devU,numBytes,cudaMemcpyDeviceToHost);
	cudaMemcpy(pU,devpU,numBytes,cudaMemcpyDeviceToHost);
	cudaMemcpy(f,devF,numBytes,cudaMemcpyDeviceToHost);

	clock_t c2 = clock();
	cerr << "Р’СЂРµРјСЏ: " << ((double)c2)/CLOCKS_PER_SEC << "РјСЃ" << endl << "РџРѕРіСЂРµС€РЅРѕСЃС‚СЊ: " << eps[0] << ",  РєРѕР»-РІРѕ РёС‚РµСЂР°С†РёР№:" <<count << endl;;
	for(int i=0; i<1024 ; i++)
	{
		for(int j=0; j<1024 ; j++){
			cout << i*h << " " << j*h << " " << U[i*1024+j]<< "\n";
		}
	}
	return 0;
}
