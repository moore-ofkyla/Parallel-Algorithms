// Name: Kyla 
// nvcc HW1.cu -o run
/*
 What to do:
 1. Understand every line of code and be able to explain it in class.
 2. Compile, run, and play around with the code.
*/

// Include files
#include <sys/time.h> //includes functions for working with time. Specifically useful for elapsed time. 
#include <stdio.h> //includes standard input and output functions, allows printing to console. 

// Defines
#define N 10000// Length of the vector( each vector will have N elements)

// Global variables
float *A_CPU, *B_CPU, *C_CPU;  //creats three variables (pointers)
float Tolerance = 0.00000001;

// Function prototypes
void allocateMemory(); //function prototypes, contains the function name, the return type, and the number and type of parameters. 
void innitialize();
void addVectorsCPU(float*, float*, float*, int);
int  check(float*, int);
long elaspedTime(struct timeval, struct timeval);
void cleanUp();

//Allocating the memory we will be using.
//function to allocate memory. 
void allocateMemory()
{	
	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
}

//Loading values into the vectors that we will add.
void innitialize()
{
	for(int i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)(2*i);
	}
}

//Adding vectors a and b then stores result in vector c.
void addVectorsCPU(float *a, float *b, float *c, int n)
{
	for(int id = 0; id < n; id++)
	{ 
		c[id] = a[id] + b[id];
	}
}

// Checking to see if anything went wrong in the vector addition.
int check(float *c, int n)
{
	int id;
	double sum = 0.0;
	double m = n-1; // Needed the -1 because we start at 0.
	
	for(id = 0; id < n; id++)
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

//struct is used because timeval holds two values tv_sec and tv_usec. Used to store time values with microsecond percision. 
	
	long startTime = start.tv_sec * 1000000 + start.tv_usec; // In microseconds. 
//long is used over float because we don't need to introduce potential floating point errors
	long endTime = end.tv_sec * 1000000 + end.tv_usec; // In microseconds

	// Returning the total time elasped in microseconds
	return endTime - startTime;
}

//Cleaning up memory after we are finished.
void CleanUp()
{
	// Freeing host "CPU" memory.
	free(A_CPU); 
	free(B_CPU); 
	free(C_CPU);
}

int main()
{
	timeval start, end;
	
	// Allocating the memory you will need.
	allocateMemory(); //function call
	
	// Putting values in the vectors.
	innitialize();

	// Starting the timer.	
	gettimeofday(&start, NULL); //function call,passes parameters. 

	// Add the two vectors.
	addVectorsCPU(A_CPU, B_CPU ,C_CPU, N);

	// Stopping the timer.
	gettimeofday(&end, NULL);
	
	// Checking to see if all went correctly.
	if(check(C_CPU, N) == 0)
	{
		printf("\n\n Something went wrong in the vector addition\n");
	}
	else
	{
		printf("\n\n You added the two vectors correctly on the CPU");
		printf("\n The time it took was %ld microseconds", elaspedTime(start, end));
	}
	
	// Your done so cleanup your room.	
	CleanUp();	
	
	// Making sure it flushes out anything in the print buffer.
//Makes sure everything is printed right
	printf("\n"); 
	
	return(0);
}

