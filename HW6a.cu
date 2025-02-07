
// Name:Kyla
// nvcc HW6a.cu -o temp -lglut -lGL
// glut and GL are openGL libraries.
/*
 What to do:
 This code displays a simple Julia fractal using the CPU.
 Rewrite the code so that it uses the GPU to create the fractal. 
 Keep the window at 1024 by 1024.

 What did I do:
 1.Not panic
 2.Set up device. Block=1024 Grid=1024, so we get 1024 by 1024. 
 3.Allocate memory on the GPU since we want to use that. 
 4. set up device
 5.intialize pixels
 6.create cuda function to do the computations
 7. make sure to copy info from GPU to CPU


*/

// Include files
#include <stdio.h>
#include <GL/glut.h>

// Defines
#define MAXMAG 10.0 // If you grow larger than this, we assume that you have escaped.
#define MAXITERATIONS 200 // If you have not escaped after this many attempts, we assume you are not going to escape.
#define A  -0.824	//Real part of C
#define B  -0.1711	//Imaginary part of C

// Global variables
unsigned int WindowWidth = 1024;
unsigned int WindowHeight = 1024;
float *pixels; //stores pixels on CPU
float *pixelsGPU; // stores pixels on GPU

dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;

// Function prototypes
void setUpDevices();//Sets up our device we'll be using
void allocateMemory();//easily shows where memory is allocated, like HW 4 
void initialize();//intializes pixel array 
void cudaErrorCheck(const char*, int);
__global__ void calculateFractalGPU(float,float, float, float,float,unsigned int, unsigned int);
//We need to pass. 1)pixels 2)Window height 3)window width 4) x man 5) x min 6)ymax 7) y min

void display(void);
void cleanup();//easily cleans up our memory, we like a clean room in this household.


void setUpDevices()//CUDA Code
{	
	BlockSize.x = 1024;// nice since this equals max number of threads
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = 1024;//like width of the display      
	GridSize.y = 1;
	GridSize.z = 1;
}

void allocateMemory()
{
	//We need the 3 because each pixel has a red, green, and blue value.
	pixels = (float *)malloc(WindowWidth*WindowHeight*3*sizeof(float)); //allocates memory on the CPU
	cudaMalloc(&pixelsGPU,WindowWidth*WindowHeight*3*sizeof(float));//allocates memory on the GPU
}

void initialize()
{
	for(int i=0;i<WindowWidth*WindowHeight*3;i ++)//initalize pixels to be black.
	{
		pixels[i]=0.0;//(1024)(1024)(3) =total number of values in pixels[]. Each pixel is represented by 3 consecuttive elements(RGB)
	}
}

void cudaErrorCheck(const char *file, int line)//checks for errors regarding cuda code
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}


__global__ void calculateFractalGPU( float *pixels, float XMin, float XMax,float YMin,float YMax,unsigned int WindowWidth, unsigned int WindowHeight)
{
//figure out x and y. do calc then color

	int id = blockIdx.x*blockDim.x + threadIdx.x;//global index
	float x, y, stepSizeX, stepSizeY;
	float mag,tempX;
	int maxCount = MAXITERATIONS;
	float maxMag = MAXMAG;
	int count = 0;

	stepSizeX = (XMax - XMin)/((float)WindowWidth);//step size in the x-direction for each pixel
	stepSizeY = (YMax - YMin)/((float)WindowHeight);//step size in the y-direction for each pixel

	x= XMin+(stepSizeX * threadIdx.x); //where the x coord is. This start at top right corner I believe. 
	y= YMin+(stepSizeY * blockIdx.x); //where the y coord is. This start at the bottom left corner I believe. 
	
	if(id< WindowHeight*WindowWidth)//if where without our 'bound' 
	{
		mag = sqrt(x*x + y*y);;
		while (mag < maxMag && count < maxCount) 
		{	
			tempX = x; //We will be changing the x but we need its old value to find y.
			x = x*x - y*y + A;
			y = (2.0 * tempX * y) + B;
			mag = sqrt(x*x + y*y);
			count++;
		}
		

// Now, assign pixel color based on the iteration count
        int pixelIdx = id * 3; //pixelIDX is the id*3 since each pixel has 3 element assosciated with it
		//think if id =1, pixelIdx=3. so pixels[1] stores the red value, pixels[2] stores the green and pixels[3] stores the blue. 

        if (count < maxCount)
        {
            // Escaped (black color)
            pixels[pixelIdx] = 0.0f;    // Red
            pixels[pixelIdx + 1] = 0.0f; // Green
            pixels[pixelIdx + 2] = 0.0f; // Blue
        }
        else
        {
            // Not escaped (red color)
            pixels[pixelIdx] = 1.0f;    // Red
            pixels[pixelIdx + 1] = 0.0f; // Green
            pixels[pixelIdx + 2] = 0.0f; // Blue
        }
}
}


void display(void) 
{ 
	
	calculateFractalGPU<<<GridSize, BlockSize>>>(pixelsGPU, XMin, XMax, YMin, YMax, WindowWidth, WindowHeight);//calls kernal
	cudaErrorCheck(__FILE__, __LINE__);//checkinf for errors after using cuda code

	cudaMemcpy(pixels,pixelsGPU, WindowHeight*WindowWidth*3*sizeof(float),cudaMemcpyDeviceToHost);//copies memory from the GPU to the CPU
	cudaErrorCheck(__FILE__, __LINE__);//check for error again

	//Putting pixels on the screen.
	glDrawPixels(WindowWidth,WindowHeight, GL_RGB, GL_FLOAT, pixels); 
	glFlush(); 
	
}

void cleanup()
{
	free(pixels);//cleaning up pixels memory
	cudaFree(pixelsGPU);//clearing up pixelsGPU memory 
}


int main(int argc, char** argv)//main function where all the action happens 
{ 
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(WindowWidth, WindowHeight);
	glutCreateWindow("Fractals--Man--Fractals");
   	glutDisplayFunc(display);

	//calls function to set up device
	setUpDevices();

	// Allocating the memory you will need. 
	allocateMemory();
	
	// Putting values for the pixels
	initialize();

   	glutMainLoop();
	cleanup();



	
}
