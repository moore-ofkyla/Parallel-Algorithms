// Name:
// N-body simulation on all available GPUs.
// nvcc HW25.cu -o temp -lglut -lm -lGLU -lGL


/*
This is a robust N-body simulation code that dynamically detects the number of GPUs
available on the machine and runs the simulation using all of them.
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


// Lennard-Jones type function parameters
#define G 10.0f
#define H 10.0f
#define LJP  2.0
#define LJQ  4.0


#define DT 0.0001
#define RUN_TIME 1.0


// Globals
int N;
int NumberOfGpus;
float3 *P, *V, *F;
float *M;
float3 **PGPUs, **VGPUs, **FGPUs;
float **MGPUs;
float GlobeRadius, Diameter, Radius;
float Damp;
dim3 BlockSize;


// Function prototypes
void cudaErrorCheck(const char *, int);
void drawPicture();
void setup();
__global__ void getForces(float3 *, float3 *, float3 *, float *, float, float, int, int, int);
__global__ void moveBodies(float3 *, float3 *, float3 *, float *, float, float, float, int, int, int);
void nBody();
int main(int, char**);


void cudaErrorCheck(const char *file, int line) {
   cudaError_t error = cudaGetLastError();
   if (error != cudaSuccess) {
       printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
       exit(0);
   }
}


void drawPicture() {
   glClear(GL_COLOR_BUFFER_BIT);
   glClear(GL_DEPTH_BUFFER_BIT);


   // Copy data from all GPUs to the host
   for (int i = 0; i < NumberOfGpus; i++) {
       int offset = (i * N) / NumberOfGpus;
       int workload = ((i + 1) * N) / NumberOfGpus - offset;

       cudaSetDevice(i);
       cudaMemcpyAsync(&P[offset], PGPUs[i], workload * sizeof(float3), cudaMemcpyDeviceToHost);
       cudaErrorCheck(__FILE__, __LINE__);
   }
   cudaDeviceSynchronize();


   glColor3d(1.0, 1.0, 0.5);


   for (int i = 0; i < N; i++) {
       glPushMatrix();
       glTranslatef(P[i].x, P[i].y, P[i].z);
       glutSolidSphere(Radius, 20, 20);
       glPopMatrix();
   }


   glutSwapBuffers();
}


void setup() {
   float randomAngle1, randomAngle2, randomRadius, d, dx, dy, dz;
   int test;


   N = 101; // Number of particles
   cudaGetDeviceCount(&NumberOfGpus);
   if (NumberOfGpus < 1) {
       printf("\nNo GPUs found. Exiting.\n");
       exit(0);
   }
   printf("You have %d GPUs\n", NumberOfGpus);


   // Allocate arrays for GPU-specific pointers
   PGPUs = (float3 **)malloc(NumberOfGpus * sizeof(float3 *));
   VGPUs = (float3 **)malloc(NumberOfGpus * sizeof(float3 *));
   FGPUs = (float3 **)malloc(NumberOfGpus * sizeof(float3 *));
   MGPUs = (float **)malloc(NumberOfGpus * sizeof(float *));


   BlockSize.x = BLOCK_SIZE;


   Damp = 0.5;


   // Allocate host memory
   M = (float *)malloc(N * sizeof(float));
   P = (float3 *)malloc(N * sizeof(float3));
   V = (float3 *)malloc(N * sizeof(float3));
   F = (float3 *)malloc(N * sizeof(float3));


   // Particle parameters
   Diameter = pow(H / G, 1.0 / (LJQ - LJP));
   Radius = Diameter / 2.0;
   float totalVolume = float(N) * (4.0 / 3.0) * PI * Radius * Radius * Radius / 0.68;
   float totalRadius = pow(3.0 * totalVolume / (4.0 * PI), 1.0 / 3.0);
   GlobeRadius = 2.0 * totalRadius;


   // Initialize particles
   for (int i = 0; i < N; i++) {
       test = 0;
       while (test == 0) {
           randomAngle1 = ((float)rand() / (float)RAND_MAX) * 2.0 * PI;
           randomAngle2 = ((float)rand() / (float)RAND_MAX) * PI;
           randomRadius = ((float)rand() / (float)RAND_MAX) * GlobeRadius;
           P[i].x = randomRadius * cos(randomAngle1) * sin(randomAngle2);
           P[i].y = randomRadius * sin(randomAngle1) * sin(randomAngle2);
           P[i].z = randomRadius * cos(randomAngle2);


           test = 1;
           for (int j = 0; j < i; j++) {
               dx = P[i].x - P[j].x;
               dy = P[i].y - P[j].y;
               dz = P[i].z - P[j].z;
               d = sqrt(dx * dx + dy * dy + dz * dz);
               if (d < Diameter) {
                   test = 0;
                   break;
               }
           }
       }


       V[i].x = 0.0; V[i].y = 0.0; V[i].z = 0.0;
       F[i].x = 0.0; F[i].y = 0.0; F[i].z = 0.0;
       M[i] = 1.0;
   }


   // Allocate GPU memory and copy data to GPUs
   for (int i = 0; i < NumberOfGpus; i++) {
       cudaSetDevice(i);
int baseWorkload = N / NumberOfGpus;
int extra = N % NumberOfGpus;
workload = baseWorkload + (i < extra ? 1 : 0);




       cudaMalloc(&MGPUs[i], workload * sizeof(float));
       cudaMalloc(&PGPUs[i], workload * sizeof(float3));
       cudaMalloc(&VGPUs[i], workload * sizeof(float3));
       cudaMalloc(&FGPUs[i], workload * sizeof(float3));


       cudaMemcpyAsync(MGPUs[i], &M[offset], workload * sizeof(float), cudaMemcpyHostToDevice);
       cudaMemcpyAsync(PGPUs[i], &P[offset], workload * sizeof(float3), cudaMemcpyHostToDevice);
       cudaMemcpyAsync(VGPUs[i], &V[offset], workload * sizeof(float3), cudaMemcpyHostToDevice);
       cudaMemcpyAsync(FGPUs[i], &F[offset], workload * sizeof(float3), cudaMemcpyHostToDevice);
   }


   printf("\nSetup finished.\n");
}


__global__ void getForces(float3 *p, float3 *v, float3 *f, float *m, float g, float h, int offset, int workload, int n) {
   float dx, dy, dz, d, d2;
   float force_mag;


   int i = threadIdx.x + blockDim.x * blockIdx.x;


   if (i < workload) {
       int global_i = i + offset;


       f[global_i].x = 0.0f;
       f[global_i].y = 0.0f;
       f[global_i].z = 0.0f;


       for (int j = 0; j < n; j++) {
           if (global_i != j) {
               dx = p[j].x - p[global_i].x;
               dy = p[j].y - p[global_i].y;
               dz = p[j].z - p[global_i].z;
               d2 = dx * dx + dy * dy + dz * dz;
               d = sqrt(d2);


               force_mag = (g * m[global_i] * m[j]) / (d2) - (h * m[global_i] * m[j]) / (d2 * d2);
               f[global_i].x += force_mag * dx / d;
               f[global_i].y += force_mag * dy / d;
               f[global_i].z += force_mag * dz / d;
           }
       }
   }
}


__global__ void moveBodies(float3 *p, float3 *v, float3 *f, float *m, float damp, float dt, float t, int offset, int workload, int n) {
   int i = threadIdx.x + blockDim.x * blockIdx.x;


   if (i < workload) {
       int global_i = i + offset;


       if (t == 0.0f) {
           v[global_i].x += ((f[global_i].x - damp * v[global_i].x) / m[global_i]) * dt / 2.0f;
           v[global_i].y += ((f[global_i].y - damp * v[global_i].y) / m[global_i]) * dt / 2.0f;
           v[global_i].z += ((f[global_i].z - damp * v[global_i].z) / m[global_i]) * dt / 2.0f;
       } else {
           v[global_i].x += ((f[global_i].x - damp * v[global_i].x) / m[global_i]) * dt;
           v[global_i].y += ((f[global_i].y - damp * v[global_i].y) / m[global_i]) * dt;
           v[global_i].z += ((f[global_i].z - damp * v[global_i].z) / m[global_i]) * dt;
       }


       p[global_i].x += v[global_i].x * dt;
       p[global_i].y += v[global_i].y * dt;
       p[global_i].z += v[global_i].z * dt;
   }
}


void nBody() {
   float t = 0.0;
   int drawCount = 0;


   while (t < RUN_TIME) {
       // Launch kernels on all GPUs
       for (int i = 0; i < NumberOfGpus; i++) {
           cudaSetDevice(i);


           int offset = (i * N) / NumberOfGpus;
           int workload = ((i + 1) * N) / NumberOfGpus - offset;


           dim3 GridSize((workload + BlockSize.x - 1) / BlockSize.x);


           getForces<<<GridSize, BlockSize>>>(PGPUs[i], VGPUs[i], FGPUs[i], MGPUs[i], G, H, offset, workload, N);
           cudaErrorCheck(__FILE__, __LINE__);


           moveBodies<<<GridSize, BlockSize>>>(PGPUs[i], VGPUs[i], FGPUs[i], MGPUs[i], Damp, DT, t, offset, workload, N);
           cudaErrorCheck(__FILE__, __LINE__);
       }


       // Synchronize GPUs
       for (int i = 0; i < NumberOfGpus; i++) {
           cudaSetDevice(i);
           cudaDeviceSynchronize();
       }


       // Perform inter-GPU communication
       for (int i = 0; i < NumberOfGpus; i++) {
           int nextGpu = (i + 1) % NumberOfGpus;
           int offset = (i * N) / NumberOfGpus;
           int workload = ((i + 1) * N) / NumberOfGpus - offset;


           cudaSetDevice(i);
           cudaMemcpyAsync(PGPUs[nextGpu], PGPUs[i], workload * sizeof(float3), cudaMemcpyDeviceToDevice);
           cudaErrorCheck(__FILE__, __LINE__);
       }


       // Synchronize GPUs after communication
       for (int i = 0; i < NumberOfGpus; i++) {
           cudaSetDevice(i);
           cudaDeviceSynchronize();
       }


       // Draw the simulation
       if (drawCount == DRAW_RATE) {
           drawPicture();
           drawCount = 0;
       }


       t += DT;
       drawCount++;
   }
}


int main(int argc, char** argv) {
   setup();


   int XWindowSize = 1000;
   int YWindowSize = 1000;


   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
   glutInitWindowSize(XWindowSize, YWindowSize);
   glutInitWindowPosition(0, 0);
   glutCreateWindow("N-body Simulation on Multiple GPUs");
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


   float3 eye = {0.0f, 0.0f, 2.0f * GlobeRadius};
   float near = 0.2;
   float far = 5.0 * GlobeRadius;


   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   glFrustum(-0.2, 0.2, -0.2, 0.2, near, far);
   glMatrixMode(GL_MODELVIEW);
   glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
   gluLookAt(eye.x, eye.y, eye.z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);


   glutMainLoop();
   return 0;
}
