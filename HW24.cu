// Name: Kyla 
// nBody code on multiple GPUs. 
// nvcc HW24.cu -o temp -lglut -lm -lGLU -lGL

/*
 What to do:
 This is some robust N-body code with all the bells and whistles removed. 
 Modify it so it runs on two GPUs.
*/

// Include files
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Defines
#define BLOCK_SIZE 128
#define PI 3.14159265359
#define DRAW_RATE 10

// This is to create a Lennard-Jones type function G/(r^p) - H(r^q). (p < q) p has to be less than q.
// In this code we will keep it a p = 2 and q = 4 problem. The diameter of a body is found using the general
// case so it will be more robust but in the code leaving it as a set 2, 4 problem make the coding much easier.
#define G 10.0f
#define H 10.0f
#define LJP  2.0
#define LJQ  4.0

#define DT 0.0001
#define RUN_TIME 10.0

// Globals
int N;
float3 *P, *V, *F;
float *M; 
float3 *PGPU1, *VGPU1, *FGPU1;
float3 *PGPU2, *VGPU2, *FGPU2;
float3 *PFromGPU1, *PFromGPU2;
float *MGPU1;
float *MGPU2;
float GlobeRadius, Diameter, Radius;
float Damp;
dim3 BlockSize;
dim3 GridSize;
dim3 GridSize2;
int N1,N2;

// Function prototypes
void cudaErrorCheck(const char *, int);
void drawPicture();
void checkForGPUs();
void setup();
__global__ void getForces(float3 *, float3 *, float3 *, float *, float, float, int, float3 *, int);
__global__ void moveBodies(float3 *, float3 *, float3 *, float *, float, float, float, int);
void nBody();
int main(int, char**);
void free();

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

void drawPicture()
{
	int i;
	
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	//cudaMemcpy vs cudaMemcpyAsync
	cudaSetDevice(0);
    cudaMemcpy(P, PGPU1, N1 * sizeof(float3), cudaMemcpyDeviceToHost);
    cudaErrorCheck(__FILE__, __LINE__);

    cudaSetDevice(1);
    cudaMemcpy(P + N1, PGPU2, N2 * sizeof(float3), cudaMemcpyDeviceToHost);
    cudaErrorCheck(__FILE__, __LINE__);

	
	glColor3d(1.0,1.0,0.5);
	for(i=0; i<N; i++)
	{
		glPushMatrix();
		glTranslatef(P[i].x, P[i].y, P[i].z);
		glutSolidSphere(Radius,20,20);
		glPopMatrix();
	}
	
	glutSwapBuffers();
}

void checkForGPUs()
{
	int deviceCount;
	cudaErrorCheck(__FILE__, __LINE__);

	cudaGetDeviceCount(&deviceCount);
	
	if(deviceCount < 2)
	{
		printf("\n\n You do not have enough GPUs to run this code(the minimum is 2).Send me your venmo\n");
		exit(0);
	}
	else
	{
		printf("\n\n You have %d GPUs available to use.\n", deviceCount);
	}

	int canAccessPeer = 0;
	cudaDeviceCanAccessPeer(&canAccessPeer, 0, 1); 
	if (canAccessPeer)
	{
		cudaSetDevice(0);
		cudaDeviceEnablePeerAccess(1, 0); // Enable P2P access from GPU 0 to GPU 1
		cudaSetDevice(1);
		cudaDeviceEnablePeerAccess(0, 0); // Enable P2P access from GPU 1 to GPU 0
		printf("P2P communication enabled between GPU 0 and GPU 1.\n");
	}
	else
	{
		printf("P2P communication not supported between GPU 0 and GPU 1.\n");
		exit(0); // Exit if P2P is required but not supported
	}
}

void setup()
{
    	float randomAngle1, randomAngle2, randomRadius;
    	float d, dx, dy, dz;
    	int test;
    	
    	N = 1000;
		N1=N/2;
		N2=N-N1;
			
		BlockSize.x = BLOCK_SIZE;
		BlockSize.y = 1;
		BlockSize.z = 1;
		
		GridSize.x = (N - 1)/BlockSize.x + 1; 
		GridSize.y = 1;
		GridSize.z = 1;

		GridSize2.x = (N2 - 1) / BlockSize.x + 1; 
		GridSize2.y = 1;
		GridSize2.z = 1;
    	Damp = 0.5;
    	
    	M = (float*)malloc(N*sizeof(float));
    	P = (float3*)malloc(N*sizeof(float3));
    	V = (float3*)malloc(N*sizeof(float3));
    	F = (float3*)malloc(N*sizeof(float3));
		//HostFromGPU1 = (float3 *)malloc(N1 * sizeof(float3)); // Buffer for positions from GPU 0
		//HostFromGPU2 = (float3 *)malloc(N2 * sizeof(float3)); // Buffer for positions from GPU 1
    
	cudaSetDevice(0);	
    cudaMalloc(&MGPU1,N1*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&PGPU1,N1*sizeof(float3));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&VGPU1,N1*sizeof(float3));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&FGPU1,N1*sizeof(float3));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&PFromGPU2, N2 * sizeof(float3)); // Buffer on GPU 0 for positions from GPU 1
	cudaErrorCheck(__FILE__, __LINE__);


	cudaSetDevice(1);
	cudaMalloc(&MGPU2,N2*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&PGPU2,N2*sizeof(float3));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&VGPU2,N2*sizeof(float3));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&FGPU2,N2*sizeof(float3));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&PFromGPU1, N1 * sizeof(float3)); // Buffer on GPU 1 for positions from GPU 0
	cudaErrorCheck(__FILE__, __LINE__);
    	
	Diameter = pow(H/G, 1.0/(LJQ - LJP)); // This is the value where the force is zero for the L-J type force.
	Radius = Diameter/2.0;
	
	// Using the radius of a body and a 68% packing ratio to find the radius of a global sphere that should hold all the bodies.
	// Then we double this radius just so we can get all the bodies setup with no problems. 
	float totalVolume = float(N)*(4.0/3.0)*PI*Radius*Radius*Radius;
	totalVolume /= 0.68;
	float totalRadius = pow(3.0*totalVolume/(4.0*PI), 1.0/3.0);
	GlobeRadius = 2.0*totalRadius;
	
	// Randomly setting these bodies in the glaobal sphere and setting the initial velosity, inotial force, and mass.
	for(int i = 0; i < N; i++)
	{
		test = 0;
		while(test == 0)
		{
			// Get random position.
			randomAngle1 = ((float)rand()/(float)RAND_MAX)*2.0*PI;
			randomAngle2 = ((float)rand()/(float)RAND_MAX)*PI;
			randomRadius = ((float)rand()/(float)RAND_MAX)*GlobeRadius;
			P[i].x = randomRadius*cos(randomAngle1)*sin(randomAngle2);
			P[i].y = randomRadius*sin(randomAngle1)*sin(randomAngle2);
			P[i].z = randomRadius*cos(randomAngle2);
			
			// Making sure the balls centers are at least a diameter apart.
			// If they are not throw these positions away and try again.
			test = 1;
			for(int j = 0; j < i; j++)
			{
				dx = P[i].x-P[j].x;
				dy = P[i].y-P[j].y;
				dz = P[i].z-P[j].z;
				d = sqrt(dx*dx + dy*dy + dz*dz);
				if(d < Diameter)
				{
					test = 0;
					break;
				}
			}
		}
	
		V[i].x = 0.0;
		V[i].y = 0.0;
		V[i].z = 0.0;
		
		F[i].x = 0.0;
		F[i].y = 0.0;
		F[i].z = 0.0;
		
		M[i] = 1.0;
	}
	
// GPU 1
cudaSetDevice(0);
cudaMemcpy(PGPU1, P, N1 * sizeof(float3), cudaMemcpyHostToDevice);
cudaMemcpy(VGPU1, V, N1 * sizeof(float3), cudaMemcpyHostToDevice);
cudaMemcpy(FGPU1, F, N1 * sizeof(float3), cudaMemcpyHostToDevice);
cudaMemcpy(MGPU1, M, N1 * sizeof(float), cudaMemcpyHostToDevice);

// GPU 2
cudaSetDevice(1);
cudaMemcpy(PGPU2, P + N1, N2 * sizeof(float3), cudaMemcpyHostToDevice);
cudaMemcpy(VGPU2, V + N1, N2 * sizeof(float3), cudaMemcpyHostToDevice);
cudaMemcpy(FGPU2, F + N1, N2 * sizeof(float3), cudaMemcpyHostToDevice);
cudaMemcpy(MGPU2, M + N1, N2 * sizeof(float), cudaMemcpyHostToDevice);
}

__global__ void getForces(float3 *p, float3 *v, float3 *f, float *m, float g, float h, int n, float3 *peerP, int peerN)
{
	float dx, dy, dz,d,d2;
	float force_mag;
	
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	
	if(i < n)
	{
		f[i].x = 0.0f;
		f[i].y = 0.0f;
		f[i].z = 0.0f;
		
		for(int j = 0; j < n; j++)
		{
			if(i != j)
			{
				dx = p[j].x-p[i].x;
				dy = p[j].y-p[i].y;
				dz = p[j].z-p[i].z;
				d2 = dx*dx + dy*dy + dz*dz;
				d  = sqrt(d2);
				
				force_mag  = (g*m[i]*m[j])/(d2) - (h*m[i]*m[j])/(d2*d2);
				f[i].x += force_mag*dx/d;
				f[i].y += force_mag*dy/d;
				f[i].z += force_mag*dz/d;
			}
		}
		 // Interactions with bodies from the other GPU
        for (int j = 0; j < peerN; j++)
        {
            dx = peerP[j].x - p[i].x;
            dy = peerP[j].y - p[i].y;
            dz = peerP[j].z - p[i].z;
            d2 = dx * dx + dy * dy + dz * dz;
            d = sqrt(d2);

            force_mag = (g * m[i]) / (d2) - (h * m[i]) / (d2 * d2);
            f[i].x += force_mag * dx / d;
            f[i].y += force_mag * dy / d;
            f[i].z += force_mag * dz / d;
        }
	}
}

__global__ void moveBodies(float3 *p, float3 *v, float3 *f, float *m, float damp, float dt, float t, int n)
{	
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	
	if(i < n)
	{
		if(t == 0.0f)
		{
			v[i].x += ((f[i].x-damp*v[i].x)/m[i])*dt/2.0f;
			v[i].y += ((f[i].y-damp*v[i].y)/m[i])*dt/2.0f;
			v[i].z += ((f[i].z-damp*v[i].z)/m[i])*dt/2.0f;
		}
		else
		{
			v[i].x += ((f[i].x-damp*v[i].x)/m[i])*dt;
			v[i].y += ((f[i].y-damp*v[i].y)/m[i])*dt;
			v[i].z += ((f[i].z-damp*v[i].z)/m[i])*dt;
		}

		p[i].x += v[i].x*dt;
		p[i].y += v[i].y*dt;
		p[i].z += v[i].z*dt;
	}
}


void free()
{
	cudaSetDevice(0);
	cudaFree(MGPU1);
	cudaFree(PGPU1);
	cudaFree(VGPU1);
	cudaFree(FGPU1);
	cudaFree(PFromGPU2);


	cudaSetDevice(1);
	cudaFree(MGPU2);
	cudaFree(PGPU2);
	cudaFree(VGPU2);
	cudaFree(FGPU2);
	cudaFree(PFromGPU1);

	free(M);
	free(P);
	free(V);
	free(F);
	
}

void nBody()
{
    int drawCount = 0; 
    float t = 0.0;
    float dt = 0.0001;

    while (t < RUN_TIME)
    {
        
		cudaMemcpyPeer(PFromGPU1, 1, PGPU1, 0, N1 * sizeof(float3));
		cudaErrorCheck(__FILE__, __LINE__);
		cudaMemcpyPeer(PFromGPU2, 0, PGPU2, 1, N2 * sizeof(float3));
		cudaErrorCheck(__FILE__, __LINE__);
		// cudaSetDevice(0);
		// cudaMemcpy(PFromGPU1, PGPU1, N1 * sizeof(float3), cudaMemcpyHostToDevice);
		// cudaErrorCheck(__FILE__, __LINE__);
		// cudaSetDevice(1);
		// cudaMemcpy( PFromGPU2, PGPU2, N2 * sizeof(float3), cudaMemcpyHostToDevice);
		// cudaErrorCheck(__FILE__, __LINE__);

        // GPU 0
        cudaSetDevice(0);
        getForces<<<GridSize, BlockSize>>>(PGPU1, VGPU1, FGPU1, MGPU1, G, H, N1, PFromGPU2, N2);
        cudaErrorCheck(__FILE__, __LINE__);
        moveBodies<<<GridSize, BlockSize>>>(PGPU1, VGPU1, FGPU1, MGPU1, Damp, DT, t, N1);
        cudaErrorCheck(__FILE__, __LINE__);
        cudaDeviceSynchronize();

        // GPU 1
        cudaSetDevice(1);
        getForces<<<GridSize2, BlockSize>>>(PGPU2, VGPU2, FGPU2, MGPU2, G, H, N2, PFromGPU1, N1);
        cudaErrorCheck(__FILE__, __LINE__);
        moveBodies<<<GridSize2, BlockSize>>>(PGPU2, VGPU2, FGPU2, MGPU2, Damp, DT, t, N2);
        cudaErrorCheck(__FILE__, __LINE__);
        cudaDeviceSynchronize();

        // Draw the picture at the specified rate
        if (drawCount == DRAW_RATE) 
        {	
            drawPicture();
            drawCount = 0;
        }

        t += dt;
        drawCount++;
    }
}
int main(int argc, char** argv)
{
	checkForGPUs();
	setup();
	
	int XWindowSize = 1000;
	int YWindowSize = 1000;
	
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("Nbody Two GPUs");
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	glutDisplayFunc(drawPicture);
	glutIdleFunc(nBody);
	
	float3 eye = {0.0f, 0.0f, 2.0f*GlobeRadius};
	float near = 0.2;
	float far = 5.0*GlobeRadius;
	
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-0.2, 0.2, -0.2, 0.2, near, far);
	glMatrixMode(GL_MODELVIEW);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluLookAt(eye.x, eye.y, eye.z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	
	glutMainLoop();
	free();
	return 0;
}

