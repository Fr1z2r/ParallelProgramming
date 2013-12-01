#include <iostream>
#include <fstream>
#include <cmath>

#define N 512
#define THREADS 32
#define BLOCKS 16
#define eps 0.005

using namespace std;


double *devU,*devU_new,*devF;
double *u,*u_new,*f;
double h;
int numberOfBytes;
	
	
double f1(int i,int j)
{
	double x=(double)i*h;
	double y=(double)j*h;
	return 4.0+2.0*x*x-2.0*x+2.0*y*y-2.0*y;
}

__global__ void relax_cu(double *u,double *f,double h2,double relax_param)
{
int i = blockIdx.x*blockDim.x + threadIdx.x;
int j = blockIdx.y*blockDim.y + threadIdx.y;
if(i>0 && i<N-1 &&j>0 && j<N-1)
            u[i*N+j]=relax_param*(u[(i-1)*N+j]+u[(i+1)*N+j]+u[i*N+j-1]+u[i*N+j+1]-h2*f[i*N+j])+(1-relax_param*4)*u[i*N+j];
}

__global__ void relax_cu2(double *u,double *u_new,double *f,double h2,double relax_param)
{
int i = blockIdx.x*blockDim.x + threadIdx.x;
int j = blockIdx.y*blockDim.y + threadIdx.y;
if(i>0 && i<N-1 &&j>0 && j<N-1)
            u_new[i*N+j]=relax_param*(u_new[(i-1)*N+j]+u[(i+1)*N+j]+u_new[i*N+j-1]+u[i*N+j+1]-h2*f[i*N+j])+(1-relax_param*4)*u[i*N+j];
}

__global__ void jacobi_cu(double *cu,double *cu_new,double *f,double h2)
{
int i = blockIdx.x*blockDim.x + threadIdx.x;
int j = blockIdx.y*blockDim.y + threadIdx.y;
if(i>0 && i<N-1 &&j>0 && j<N-1)
            cu_new[i*N+j]=0.25*(cu[(i-1)*N+j]+cu[(i+1)*N+j]+cu[i*N+j-1]+cu[i*N+j+1]-h2*f[i*N+j]);
}

void err_cuda(cudaError_t i)
{
if(i!=0){
	printf("Error: %i\t%s\n",i,cudaGetErrorString(i));
	exit(-1);
}
}

void cudaInit()
{
	err_cuda(cudaMalloc((void**)&devU,numberOfBytes));
	err_cuda(cudaMalloc((void**)&devU_new,numberOfBytes));
	err_cuda(cudaMalloc((void**)&devF,numberOfBytes));
	err_cuda(cudaMemcpy(devU,u,numberOfBytes,cudaMemcpyHostToDevice));
	err_cuda(cudaMemcpy(devF,f,numberOfBytes,cudaMemcpyHostToDevice));
	err_cuda(cudaMemcpy(devU_new,u_new,numberOfBytes,cudaMemcpyHostToDevice));
}

void cudaFinal()
{
	err_cuda(cudaFree(devU));
	err_cuda(cudaFree(devU_new));
	err_cuda(cudaFree(devF));
	
}

void init()
{	
	h=1.0/N;
	numberOfBytes=sizeof(double)*N*N;
	u=new double[numberOfBytes];
	u_new=new double[numberOfBytes];
	f=new double[numberOfBytes];
	
	for(int i=0;i<N;i++)
	{
		for(int j=0;j<N;j++)
		{
			u[i*N+j]=0;
			u_new[i*N+j]=0;
			f[i*N+j]=f1(i,j);
		}
	}
	
	double tmp;
	for(int i=0;i<N;i++)
	{
		tmp=(double)i*h*(double)i*h-(double)i*h+1;
		u[i*N]=tmp;
		u[i*N+N-1]=tmp;
		u[i]=tmp;
		u[(N-1)*N+i]=tmp;
		
		u_new[i*N]=tmp;
		u_new[i*N+N-1]=tmp;
		u_new[i]=tmp;
		u_new[(N-1)*N+i]=tmp;
	}
}

void final()
{
	delete[] u;
	delete[] u_new;
	delete[] f;
}

double exact_solve(int i,int j)
{
	double x=(double)i*h;
	double y=(double)j*h;
	double res=(x*x-x+1.0)*(y*y-y+1.0);
	return res;
}

bool err()
{
	double max=0;
	double tmp;
	for(int i=1;i<N-1;i++)
		for(int j=1;j<N-1;j++)
		{
			tmp=fabs(u[i*N+j]-exact_solve(i,j));
			if(tmp>max)max=tmp;
		}
		cout<<"err: "<<max<<endl;
		if(max<eps)return true;
		return false;

}

void writeToFile(string fname)
{
	ofstream output_file;
	output_file.open(fname.c_str(),std::ofstream::out);
	
	for(int i=0;i<N;i++)
		for(int j=0;j<N;j++)
		{
			output_file<<(double)i*h <<" "<<(double)j*h<<" "<<u[i*N+j]<<endl;
		}
}

void jacobi(double param)
{
	dim3 threads = dim3(THREADS, THREADS,1);
	dim3 blocks  = dim3(BLOCKS, BLOCKS,1);
	
	for(int i=0;;i++)
	{
    	    jacobi_cu<<<threads,blocks,0>>>(devU,devU_new,devF,h*h);
			double *tmp=devU;
			devU=devU_new;
			devU_new=tmp;
			if(i%10000==0)
			{
				err_cuda(cudaMemcpy(u,devU,numberOfBytes,cudaMemcpyDeviceToHost));				
				cout<<i<<endl;
				if(err())break;
			}
			
	}
}

void relax(double param)
{
	dim3 threads = dim3(THREADS, THREADS,1);
	dim3 blocks  = dim3(BLOCKS, BLOCKS,1);
	
	for(int i=0;;i++)
	{
    	    relax_cu<<<threads,blocks,0>>>(devU,devF,h*h,param);			
			if(i%100==0)
			{
				cout<<i<<endl;
				err_cuda(cudaMemcpy(u,devU,numberOfBytes,cudaMemcpyDeviceToHost));
				
				if(err())break;
			}
			
	}
}

void relax2(double param)
{
	dim3 threads = dim3(THREADS, THREADS,1);
	dim3 blocks  = dim3(BLOCKS, BLOCKS,1);
	
	for(int i=0;;i++)
	{
    	    relax_cu2<<<threads,blocks,0>>>(devU,devU_new,devF,h*h,param);			
			if(i%10000==0)
			{
				double *tmp=devU;
				devU=devU_new;
				devU_new=tmp;
				cout<<i<<endl;
				err_cuda(cudaMemcpy(u,devU,numberOfBytes,cudaMemcpyDeviceToHost));				
				if(err())break;
			}
			
	}
}

double CalcTimeOfMethod(void (*method)(double),double param)
{
	init();
	cudaInit();	
	
	clock_t start=clock();
	method(param);
	double time=(double)(clock()-start)/CLOCKS_PER_SEC;
	
	//writeToFile("output2.dat");
	
	cudaFinal();
	final();
	return time;
}

int main(int argc, char* argv[])
{	
	double t1=CalcTimeOfMethod(&jacobi,0.25);
	double t2=CalcTimeOfMethod(&relax,0.25);
	double t3=CalcTimeOfMethod(&relax,0.18);
	double t4=CalcTimeOfMethod(&relax,0.33);
	cout<<"Time: "<<t1<<endl;
	cout<<"Time: "<<t2<<endl;
	cout<<"Time: "<<t3<<endl;
	cout<<"Time: "<<t4<<endl;
	return 0;
	
}