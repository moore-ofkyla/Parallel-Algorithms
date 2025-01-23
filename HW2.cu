// Name:Kyla
// nvcc HW2.cu -o run
/*
 What to do:
 This code adds the vectors on the GPU.
 Man, that was easy!

 1. First, just add cuda to the word malloc to get cudaMalloc and use it to allocate memory on the GPU.
 Okay, you had to use an & instead of float*, but come on, that was no big deal.

 2. Use cudaMemcpyAsync to copy your CPU memory holding your vectors to the GPU.

 3. Now for the important stuff we've all been waiting for: the GPU "CUDA kernel" that does 
 the work on thousands of CUDA cores all at the same time!!!!!!!! 
 Wait, all you have to do is remove the for loop?
 Dude, that was too simple! I want my money back! 
 Be patient, it gets a little harder, but remember, I told you CUDA was simple.
 
 4. call cudaDeviceSynchronize. SYnc up the CPU and the GPU. I'll expaned on this in to story at the end of 5 below.
 
 5. Use cudaMemcpyAsync again to copy your GPU memory back to the CPU.
 Be careful with cudaMemcpyAsync. Make sure you pay attention to the last argument you pass in the call.
 Also, note that it says "Async" at the end. That means the CPU tells the GPU to do the copy but doesn't wait around for it to finish.

 CPU: "Dude, do your copy and don't bother me. It's 'Async'—I’ve got to get back to watching this cool 
 TikTok video of a guy smashing watermelons with his face."
 
 GPU: "Whatever, dude. I'll do your copy when I get around to it. It's 'Async'."
 
 CPU: "Just make sure you get it done before I check your work."
 
 GPU: "Well, maybe you'd better check with me to see if I'm done before you start checking. That means use cudaDeviceSynchronize!"
 
 CPU: "Da."
 
 GPU: "I might be all tied up watching a TikTok video of a guy eating hotdogs with his hands tied behind his back... underwater."
 
 GPU thought to self: "It must be nice being a CPU, living in the administration zone where time and logic don't apply. 
 Sitting in meetings all day coming up with work for us to do!"

 6. Use cudaFree instead of free.
 
 What you need to do:

 The code below runs for a vector of length 500.
 Modify it so that it runs for a vector of length 1000 and check your result.
 Then, set the vector size to 1500 and check your result again. 
 This is the code you will turn in.
 
 Remember, you can only use one block!!!
 Don’t cry. I know you played with a basket full of blocks when you were a kid.
 I’ll let you play with over 60,000 blocks in the future—you’ll just have to wait.

 Be prepared to explain what you did to make this work and why it works.
 NOTE: Good code should work for any value of N.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
#define N 1500
// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float Tolerance = 0.00000001;

// Function prototypes
void setUpDevices();
void allocateMemory();
void innitialize();
void addVectorsCPU(float*, float*, float*, int);
__global__ void addVectorsGPU(float, float, float, int);
int  check(float*, int);
long elaspedTime(struct timeval, struct timeval);
void cleanUp();

// This will be the layout of the parallel space we will be using.
void setUpDevices()
{
if (N>1024)// Limits size of space to 1024(one block) even if we have more than 1024 tasks. 
{
	BlockSize.x = 1024;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = 1;
	GridSize.y = 1;
	GridSize.z = 1;
}
else //just makes the space be equal to the size of N
{
	BlockSize.x = N;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = 1;
	GridSize.y = 1;
	GridSize.z = 1;
}
}

// Allocating the memory we will be using.
void allocateMemory()
{	
	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
	
	// Device "GPU" Memory
	cudaMalloc(&A_GPU,N*sizeof(float));
	cudaMalloc(&B_GPU,N*sizeof(float));
	cudaMalloc(&C_GPU,N*sizeof(float));

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
	                          
	int id= threadIdx.x ; //sets id value equal to the thread index
	int k=0; //intializes k to 0

		if(n>1024)//more calculations required than number of threads. So we need to do more than one calculations on some. 
		{
			while(id + (k * blockDim.x) < n) //runs while id + k * blocDim.x is less than n. blockdim.x is 1024 I believe, and k is the iteration 
			{
					
					c[id + k * blockDim.x] = a[id + k * blockDim.x] + b[id + k * blockDim.x];
					//calculates the value at id+ (k*blockDim.x location).
					//on the first run this will just be id since k is intially 0
					k++;//increments k by 1
					
			}
		}

		else//runs if n<1024.
		{
			c[id] = a[id] + b[id];
		}
	}


// Checking to see if anything went wrong in the vector addition.
int check(float *c, int n)
{
	double sum = 0.0;
	double m = n-1; // Needed the -1 because we start at 0.
	
	for(int id = 0; id < n; id++)
	{ 
		sum += c[id];
	}
	
	if(abs(sum - 3.0*(m*(m+1))/2.0) < Tolerance) 
	{
		return(1);
	}
	else 
	{
		return(0);
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
	
	cudaFree(A_GPU); 
	cudaFree(B_GPU); 
	cudaFree(C_GPU);
}

int main()
{
	timeval start, end;
	long timeCPU, timeGPU;
	
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
	
	// Adding on the GPU
	gettimeofday(&start, NULL);
	
	// Copy Memory from CPU to GPU		
	cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	
	addVectorsGPU<<<GridSize,BlockSize>>>(A_GPU, B_GPU ,C_GPU, N);
	
	// Copy Memory from GPU to CPU	
	cudaMemcpyAsync(C_CPU, C_GPU, N*sizeof(float), cudaMemcpyDeviceToHost);
	
	// Making sure the GPU and CPU wiat until each other are at the same place.
	cudaDeviceSynchronize();
	
	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);
	
	// Checking to see if all went correctly.
	if(check(C_CPU, N) == 0)
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
	printf("\n");
	
	return(0);
}

