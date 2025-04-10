// Name:Kyla 
// Vector addition on two GPUs.
// nvcc HW22.cu -o temp
/*
 What to do:
 This code adds two vectors of any length on a GPU.
 Rewriting the Code to Run on Two GPUs:

 1. Check GPU Availability:
    Ensure that you have at least two GPUs available. If not, report the issue and exit the program.

 2. Handle Odd-Length Vector:
    If the vector length is odd, ensure that you select a half N value that does not exclude the last element of the vector.

 3. Send First Half to GPU 1:
    Send the first half of the vector to the first GPU, and perform the operation of adding a to b.

 4. Send Second Half to GPU 2:
    Send the second half of the vector to the second GPU, and again perform the operation of adding a to b.

 5. Return Results to the CPU:
    Once both GPUs have completed their computations, transfer the results back to the CPU and verify that the results are correct.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
#define N 11503 // Length of the vector

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU1, *B_GPU1, *C_GPU1; //GPU pointers
float *A_GPU2, *B_GPU2, *C_GPU2; //GPU pointers
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float Tolerance = 0.01;

// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices();
void allocateMemory();
void checkforGPUs();
void innitialize();
void addVectorsCPU(float*, float*, float*, int);
__global__ void addVectorsGPU(float, float, float, int);
bool  check(float*, int);
long elaspedTime(struct timeval, struct timeval);
void cleanUp();

// This check to see if an error happened in your CUDA code. It tell you what it thinks went wrong,
// and what file and line it occured on.
void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}

// This will be the layout of the parallel space we will be using.
void setUpDevices()
{
	BlockSize.x = 256;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = (N - 1)/BlockSize.x + 1; // This gives us the correct number of blocks.
	GridSize.y = 1;
	GridSize.z = 1;
}
void checkForGPUs()
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	
	if(deviceCount < 2)
	{
		printf("\n\n You do not have enough GPUs to run this code. You need at least 2 GPUs.\n");
		exit(0);
	}
	else
	{
		printf("\n\n You have %d GPUs available to use.\n", deviceCount);
	}
}

// Allocating the memory we will be using.
void allocateMemory()
{	
	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
	
	int halfN1 = N / 2;
    int halfN2 = N - halfN1;

	cudaSetDevice(0);
    cudaMalloc(&A_GPU1, halfN1 * sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
    cudaMalloc(&B_GPU1, halfN1 * sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
    cudaMalloc(&C_GPU1, halfN1 * sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);


    cudaSetDevice(1);
    cudaMalloc(&A_GPU2, halfN2 * sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
    cudaMalloc(&B_GPU2, halfN2 * sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
    cudaMalloc(&C_GPU2, halfN2 * sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
}

// Loading values into the vectors that we will add.
void innitialize()
{
	for(int i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)(2*i);
	}
}

// Adding vectors a and b on the CPU then stores result in vector c.
void addVectorsCPU(float *a, float *b, float *c, int n)
{
	for(int id = 0; id < n; id++)
	{ 
		c[id] = a[id] + b[id];
	}
}

// This is the kernel. It is the function that will run on the GPU.
// It adds vectors a and b on the GPU then stores result in vector c.
__global__ void addVectorsGPU(float *a, float *b, float *c, int n)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(id < n) // Making sure we are not working on memory we do not own.
	{
		c[id] = a[id] + b[id];
	}
}

// Checking to see if anything went wrong in the vector addition.
bool check(float *c, int n, float tolerence)
{
	int id;
	double myAnswer;
	double trueAnswer;
	double percentError;
	double m = n-1; // Needed the -1 because we start at 0.
	
	myAnswer = 0.0;
	for(id = 0; id < n; id++)
	{ 
		myAnswer += c[id];
	}
	
	trueAnswer = 3.0*(m*(m+1))/2.0;
	
	percentError = abs((myAnswer - trueAnswer)/trueAnswer)*100.0;
	
	if(percentError < Tolerance) 
	{
		return(true);
	}
	else 
	{
		return(false);
	}
}

// Calculating elasped time.
long elaspedTime(struct timeval start, struct timeval end)
{
	// tv_sec = number of seconds past the Unix epoch 01/01/1970
	// tv_usec = number of microseconds past the current second.
	
	long startTime = start.tv_sec * 1000000 + start.tv_usec; // In microseconds.
	long endTime = end.tv_sec * 1000000 + end.tv_usec; // In microseconds

	// Returning the total time elasped in microseconds
	return endTime - startTime;
}

// Cleaning up memory after we are finished.
void CleanUp()
{
	// Freeing host "CPU" memory.
	free(A_CPU); 
	free(B_CPU); 
	free(C_CPU);
	
	cudaSetDevice(0);
	cudaErrorCheck(__FILE__, __LINE__);
    cudaFree(A_GPU1);
	cudaErrorCheck(__FILE__, __LINE__);
    cudaFree(B_GPU1);
	cudaErrorCheck(__FILE__, __LINE__);
    cudaFree(C_GPU1);
	cudaErrorCheck(__FILE__, __LINE__);

    cudaSetDevice(1);
	cudaErrorCheck(__FILE__, __LINE__);
    cudaFree(A_GPU2);
	cudaErrorCheck(__FILE__, __LINE__);
    cudaFree(B_GPU2);
	cudaErrorCheck(__FILE__, __LINE__);
    cudaFree(C_GPU2);
	cudaErrorCheck(__FILE__, __LINE__);
}

int main()
{
	timeval start, end;
	long timeCPU, timeGPU;
	checkForGPUs();
	// Setting up the GPU
	setUpDevices();
	
	// Allocating the memory you will need.
	allocateMemory();
	
	// Putting values in the vectors.
	innitialize();
	
	// Adding on the CPU
	gettimeofday(&start, NULL);
	addVectorsCPU(A_CPU, B_CPU ,C_CPU, N);
	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);
	
	// Zeroing out the C_CPU vector just to be safe because right now it has the correct answer in it.
	for(int id = 0; id < N; id++)
	{ 
		C_CPU[id] = 0.0;
	}
	int halfN1 = N / 2;
    int halfN2 = N - halfN1;
	// Adding on the GPU
	gettimeofday(&start, NULL);
	
	//select 1st GPU send up the info for that one
	cudaSetDevice(0);
    cudaMemcpy(A_GPU1, A_CPU, halfN1 * sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
    cudaMemcpy(B_GPU1, B_CPU, halfN1 * sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);

	//select 2nd GPU send up the info for that one
    cudaSetDevice(1);
    cudaMemcpy(A_GPU2, A_CPU + halfN1, halfN2 * sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
    cudaMemcpy(B_GPU2, B_CPU + halfN1, halfN2 * sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);


    // Launch kernels on both GPUs
    cudaSetDevice(0);
    addVectorsGPU<<<(halfN1 + BlockSize.x - 1) / BlockSize.x, BlockSize>>>(A_GPU1, B_GPU1, C_GPU1, halfN1);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);

    cudaSetDevice(1);
    addVectorsGPU<<<(halfN2 + BlockSize.x - 1) / BlockSize.x, BlockSize>>>(A_GPU2, B_GPU2, C_GPU2, halfN2);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);
	

    // Copy results back to CPU
	//SELECT 1st GPU, then copy the results back
    cudaSetDevice(0);
    cudaMemcpy(C_CPU, C_GPU1, halfN1 * sizeof(float), cudaMemcpyDeviceToHost);
	//select 2nd GPU, then copy the results back
    cudaSetDevice(1);
    cudaMemcpy(C_CPU + halfN1, C_GPU2, halfN2 * sizeof(float), cudaMemcpyDeviceToHost);

	
	// Making sure the GPU and CPU wiat until each other are at the same place.
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);
	
	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);
	
	// Checking to see if all went correctly.
	if(check(C_CPU, N, Tolerance) == false)
	{
		printf("\n\n Something went wrong in the GPU vector addition\n");
	}
	else
	{
		printf("\n\n You added the two vectors correctly on the GPU");
		printf("\n The time it took on the CPU was %ld microseconds", timeCPU);
		printf("\n The time it took on the GPU was %ld microseconds", timeGPU);
	}
	
	// Your done so cleanup your room.	
	CleanUp();	
	
	// Making sure it flushes out anything in the print buffer.
	printf("\n\n");
	
	return(0);
}
