// Name:Kyla Moore
// nvcc HW5.cu -o temp
/*
 What to do:
 This code prints out useful information about the GPU(s) in your machine, 
 but there is much more data available in the cudaDeviceProp structure.


 Extend this code so that it prints out all the information about the GPU(s) in your system. 
 Also, and this is the fun part, be prepared to explain what each piece of information means. 
*/

// Include files
#include <stdio.h>

// Defines

// Global variables

// Function prototypes
void cudaErrorCheck(const char*, int);

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

int main()
{
	cudaDeviceProp prop;

	int count;
	cudaGetDeviceCount(&count);
	cudaErrorCheck(__FILE__, __LINE__);
	printf(" You have %d GPU(s) in this machine\n", count);
	
	for (int i=0; i < count; i++) {
		cudaGetDeviceProperties(&prop, i);
		cudaErrorCheck(__FILE__, __LINE__);

		printf(" ---General Information for device %d ---\n", i);//device information stored below. 
		printf("Name: %s\n", prop.name); // name of device. (I)
		printf("UUID:");//16-byte identifier that uniquely identifies the GPU device.
		for (int j = 0; j < 16; j++) {
    printf("%02x", (unsigned char)prop.uuid.bytes[j]);
}
	printf("\n");

	printf("luid: ");//8-byte locally unique identifier. Value is undefined on TCC and non-Windows platforms 
    for (int k = 0; k < 8; k++) {
        printf("%02x", static_cast<unsigned char>(prop.luid[k]));
        if (k < 7) printf(":"); // Add colon separator between bytes
    }
	printf("\n");
		printf("luidDeviceNodeMask: %u \n", prop.luidDeviceNodeMask);// LUID device node mask. Value is undefined on TCC and non-Windows platforms
		printf("Compute capability: %d.%d\n", prop.major, prop.minor); // major and minor compute ability
		printf("Clock rate: %d\n", prop.clockRate);//clock frequency in kilohertz
		printf("Device copy overlap: ");
		if (prop.deviceOverlap) printf("Enabled\n");//Device can concurrently copy memory and execute a kernel.
		else printf("Disabled\n");
		printf("Kernel execution timeout : ");
		if (prop.kernelExecTimeoutEnabled) printf("Enabled\n");// Specified whether there is a run time limit on kernels
		else printf("Disabled\n");
		printf(" ---Memory Information for device %d ---\n", i);
		printf("Total global mem: %ld\n", prop.totalGlobalMem);//global memory available in bytes
		printf("Total constant Mem: %ld\n", prop.totalConstMem);//constant memory available in bytes
		printf("Max mem pitch: %ld\n", prop.memPitch);// Maximum pitch in bytes allowed by memory copies 
		printf("Texture Alignment: %ld\n", prop.textureAlignment);//specifies the minimum memory alignment (in bytes) required for textures to be accessed efficiently by the GPU 
		printf("Texture Pitch Alignment %ld\n",prop.texturePitchAlignment);//defines the alignment (in bytes) required for the pitch (row width) of 2D texture memory
		printf(" ---MP Information for device %d ---\n", i);
		printf("Multiprocessor count : %d\n", prop.multiProcessorCount); //Number of SMs on the GPU
		printf("kernelExecTimeoutEnabled: %d\n", prop.kernelExecTimeoutEnabled);//Specified whether there is a run time limit on kernels
		printf("Shared mem per mp: %ld\n", prop.sharedMemPerBlock);//shared memory reserved by CUDA driver per block in bytes
		printf("Registers per mp: %d\n", prop.regsPerBlock);//32-bit registers available per block
		printf("Threads in warp: %d\n", prop.warpSize);//the number of threads that are processed simultaneously by a single Streaming Multiprocessor 
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);//you'll never guess. Max threads per block!
		printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);//max dimension of each dim of a block

		printf("\n");
		printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);//max size of each dim of a grid
		printf("Integrated: %s \n",prop.integrated ? "Yes" : "No");//ndicates whether the GPU is integrated (sharing memory with the CPU) or discrete (having its own dedicated memory)
		printf("Can Map Host Memory: %s \n",prop.canMapHostMemory ? "Yes" : "No"); //checks if the GPU can directly access memory that is in the CPU memory space.
 		printf("computeMode: %d\n", prop.computeMode);// 0 means=can run multiple kernals concurrently, 1=can only run one kernal at a time. 2=the devide cant be used for computation
		
		printf("\n");
		printf("Max 1D Texture: %d\n",prop.maxTexture1D);//This field tells you the maximum size (in elements) of a 1D texture that the GPU can handle. 
		printf("Max Texture 1D Mipmap: %d\n", prop.maxTexture1DMipmap); //This indicates the maximum size of a 1D texture that supports mipmaps.
		printf("Max Texture 1D Linear: %d\n", prop.maxTexture1DLinear); //This specifies the maximum size of a 1D texture in linear memory layout.
		printf("Max Texture 1D Layered(size, layers): (%d,%d)\n", prop.maxTexture1DLayered[0],prop.maxTexture1DLayered[1]); //It tells you the largest size of each individual 1D texture layer and how many layers of 1D textures the GPU can handle in a single 1D layered texture.
		printf("Max Texture 2D:(%d,%d)\n",prop.maxTexture2D[0],prop.maxTexture2D[1]);//max widith,max height 
		printf("Max Texture 2D Mipmap: (%d, %d)\n", prop.maxTexture2DMipmap[0], prop.maxTexture2DMipmap[1]);//tells max width and height of a 2d texture with mipmaps((precomputed image pyramids used to improve performance)
		printf("Max Texture 2D Linear: (%d, %d, %d)\n", prop.maxTexture2DLinear[0], prop.maxTexture2DLinear[1], prop.maxTexture2DLinear[2]);//describes the max size of a 2D texture with linear memory layout
		printf("Max Texture 2D Gather: (%d, %d)\n", prop.maxTexture2DGather[0], prop.maxTexture2DGather[1]);//describes the maximum size of a 2D texture that supports gather operations( a way of sampling data)
		printf("Max Texture 3D: (%d, %d, %d)\n", prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]);//max dimensions of a 3D texture(width,height and depth)
		printf("Max Texture 3D Alt: (%d, %d, %d)\n", prop.maxTexture3DAlt[0], prop.maxTexture3DAlt[1], prop.maxTexture3DAlt[2]);//similar to maxTexture 3D, but provides alternative limits for 3D texturese
		printf("Max Texture Cubemap: %d\n", prop.maxTextureCubemap);//describes the max size of a cubemap texture. A cubemap is 6 2D textures, one for each face of a cube. 
		printf("Max Texture 2D Layered: (%d, %d, %d)\n", prop.maxTexture2DLayered[0], prop.maxTexture2DLayered[1], prop.maxTexture2DLayered[2]);//max size of a 2D texture that can have multiple layers
		printf("Max Texture Cubemap Layered: (%d, %d)\n", prop.maxTextureCubemapLayered[0], prop.maxTextureCubemapLayered[1]);//specifices the max size of a cubemap and the number of layers
		printf("Max Surface 1D: %d\n", prop.maxSurface1D);//max size of a 1D surface. One-dimensional texture or image that can hold data in a single row.
		printf("Max Surface 2D: (%d, %d)\n", prop.maxSurface2D[0], prop.maxSurface2D[1]);//max dimensions of a 2D surface
		printf("Max Surface 3D: (%d, %d, %d)\n", prop.maxSurface3D[0], prop.maxSurface3D[1], prop.maxSurface3D[2]);//Max diminsions of a 3D surface
		printf("Max Surface 1D Layered: (%d, %d)\n", prop.maxSurface1DLayered[0], prop.maxSurface1DLayered[1]);//Maximum size and number of layers for a 1D layered surface.
		printf("Max Surface 2D Layered: (%d, %d, %d)\n", prop.maxSurface2DLayered[0], prop.maxSurface2DLayered[1], prop.maxSurface2DLayered[2]);//The max size of a 2D texture with multiple layers		
		printf("Max Surface Cubemap: %d\n", prop.maxSurfaceCubemap);//Represents the maximum size of a cubemap texture 
		printf("Max Surface Cubemap Layered: (%d, %d)\n", prop.maxSurfaceCubemapLayered[0], prop.maxSurfaceCubemapLayered[1]);// gives the max size and number of layers that a cubemap can have
		printf("\n");
		printf("Surface Alignment: %ld bytes\n", prop.surfaceAlignment);//tells you how memory should be aligned for textures to work properly.
		printf("Concurrent Kernels: %s\n", prop.concurrentKernels ? "Yes" : "No");//Tells you if a GPU can execute multiple kernals concurrently(parallel)
		printf("ECC Enabled: %s\n", prop.ECCEnabled ? "Yes" : "No");//EEC helps detect errors and correct them in the memory of the GPU
		printf("PCI Bus ID: %d\n", prop.pciBusID);//The ID of the bus that connects the GPU to the rest of the computer's system.
		printf("PCI Device ID: %d\n", prop.pciDeviceID);//the unique ID for the GPU on that PCI bus
		printf("PCI Domain ID: %d\n", prop.pciDomainID);//If there are multiple PCI domains(group of buses), this tells you the domain where the GPU is located, 
		printf("TCC Driver Mode: %s\n", prop.tccDriver ? "Yes" : "No");// TCC(only for computing tasks no display output) or WDDM mode (normal graphic tasks)
		printf("Asynchronous Engine Count: %d\n", prop.asyncEngineCount);//async engines allow the GPU to handle multiple tasks at the same time.
		printf("Unified Addressing: %s\n", prop.unifiedAddressing ? "Yes" : "No");//Allows the CPU and GPU to use the same memory space for easier data sharing.
		printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);// widith of bus memory
		printf("Memory Clock Rate: %d kHz\n", prop.memoryClockRate);//gives memory clock speed in kHz(kilohertz)
		printf("L2 Cache Size: %d bytes\n", prop.l2CacheSize);//Size of L2 cache in bytes 
		printf("Persisting L2 Cache Max Size: %d bytes\n", prop.persistingL2CacheMaxSize);//max size of L2 cache that can be persisted across kernal launches. means the GPU keeps useful data in the L2 cache even between different tasks
		printf("Max Threads Per MultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor);//max number of thread that can be processed at the same time on ONE multiprocessor 
		printf("Stream Priorities Supported: %s\n", prop.streamPrioritiesSupported ? "Yes" : "No");//Is stream priorities supported(streams are operations executed in order,prioties allow you to control the order streams are executed)
		printf("Global L1 Cache Supported: %s\n", prop.globalL1CacheSupported ? "Yes" : "No");//Shared across all cores. Small,fast memeory location on EACH Streaming multiprocessor 
		printf("Local L1 Cache Supported: %s\n", prop.localL1CacheSupported ? "Yes" : "No");//supports L1 Cache,small,fast memory close to the processor cores.Specific to an indivual core
		printf("Shared Memory Per Multiprocessor: %ld bytes\n", prop.sharedMemPerMultiprocessor);//amount of shared memory available per multiprocessor 
		printf("Registers Per Multiprocessor: %d\n", prop.regsPerMultiprocessor);//how many registers are available per multiprocessor
		printf("Managed Memory: %s\n", prop.managedMemory ? "Yes" : "No");// Yes=CPU and GPU can share the same memory space
		printf("Can Use Host Pointer for Registered Memory: %s\n", prop.canUseHostPointerForRegisteredMem ? "Yes" : "No");//Tells you if the GPU can use host memory pointers directly in registered memory operations
		printf("Is Multi-GPU Board: %s\n", prop.isMultiGpuBoard ? "Yes" : "No");//Is there more than one GPU?
		printf("hostNativeAtomicSupported: %s\n", prop.hostNativeAtomicSupported ? "Yes" : "No");//tells you if the GPU can perform atomic operations on memory that both the CPU and GPU can access.
		printf("Multi GPU Board Group ID: %d\n", prop.multiGpuBoardGroupID);// a number that identifies which "group" of GPUs a particular GPU belongs to when you have multiple GPUs in your system.
		printf("\n");
		printf("Single to Double Precision Performance Ratio: %d\n", prop.singleToDoublePrecisionPerfRatio);//shows how much faster the GPU can perform single-percision calculations
		printf("Pageable Memory Access: %s\n", prop.pageableMemoryAccess ? "Yes" : "No");//Pageable memory is able to be swapped in and out of RAM(by the OS)
		printf("Concurrent Managed Access: %s\n",prop.concurrentManagedAccess ? "Yes" : "No");// Concurrernt Access to Memory=Both GPU and CPU can access the same region of memory,without causing an issue. 
		printf("Compute Preemption Supported: %s\n", prop.computePreemptionSupported ? "Yes" : "No");// GPU is capable of pausing execution on one kernal and start the excution of another kernal. 
		printf("Cooperative Launch: %s\n", prop.cooperativeLaunch ? "Yes" : "No");//refers to syncing multiple kernals that run on the same gpu
		printf("Cooperative Multi-Device Launch: %s\n", prop.cooperativeMultiDeviceLaunch ? "Yes" : "No");// !! yes=G multiple GPUS can work together. No=Only one GPU will be used for the task.
		printf("Pageable Memory Access Uses Host Page Tables: %s \n",prop.pageableMemoryAccessUsesHostPageTables? "Yes" : "No");//indicares if GPU can access pageable memory directly through the host’s page tables.
		printf("Direct Managed Mem Access From Host: %s \n",prop.directManagedMemAccessFromHost? "Yes" : "No");//Host can directly access managed memory on the device without migration.
		printf("Access Policy Max Window Size: %d \n",prop.accessPolicyMaxWindowSize);//represents the maximum size (in bytes) of a memory access window that can be used in CUDA applications when working with memory access policies.
		printf("Max Blocks Per Multi Processor: %d\n", prop.maxBlocksPerMultiProcessor);//Maximum number of resident blocks per multiprocessor
		printf("Reserved Shared Mem Per Block: %ld bytes\n", prop.reservedSharedMemPerBlock);//shared memory reserved by CUDA driver per block in bytes
		printf("Host Register Supported: %d\n", prop.hostRegisterSupported);//The hostRegisterSupported field indicates whether the GPU can register host memory using cudaHostRegister, allowing the memory to be pinned and directly accessed by the GPU
		printf("Sparse Cuda Array Supported: %d\n", prop.sparseCudaArraySupported);//1 if the device supports sparse CUDA arrays and sparse CUDA mipmapped arrays, 0 otherwise 
		printf("Host Register Read Only Supported: %d\n", prop.hostRegisterReadOnlySupported);//The hostRegisterReadOnlySupported field indicates whether the GPU can register host memory as read-only using the cudaHostRegister API, meaning the memory can only be accessed by the GPU for reading, not writing.
		printf("Timeline Semaphore Interop Supported: %d\n", prop.timelineSemaphoreInteropSupported);//indicates whether the GPU supports interoperability with external timeline semaphores, allowing the device to synchronize with other systems or devices that also use timeline semaphores for synchronization.
		printf("Memory Pools Supported: %d\n", prop.memoryPoolsSupported);//1 if the device supports using the cudaMallocAsync and cudaMemPool family of APIs, 0 otherwise
		printf("Gpu Direct RDMA Supported: %d\n", prop.gpuDirectRDMASupported);// 1 if the device supports GPUDirect RDMA APIs, 0 otherwise
		printf("Gpu Direct RDMA Flush Writes Options: 0x%X\n", prop.gpuDirectRDMAFlushWritesOptions);//It controls how the GPU ensures that memory writes are properly synchronized or flushed when using RDMA, based on the specific flags set in the bitmask.
		printf("Gpu Direct RDMA Writes Ordering: %d\n", prop.gpuDirectRDMAWritesOrdering);//It controls how the GPU orders memory writes when performing direct memory transfers to/from other devices via RDMA.
		printf("Memory Pool Supported Handle Types: 0x%X\n", prop.memoryPoolSupportedHandleTypes);//It shows which types of memory handle options the GPU supports when using memory pools to manage memory allocations.
		printf("Deferred Mapping Cuda Array Supported: %d\n", prop.deferredMappingCudaArraySupported);//It tells you if the GPU can delay mapping a CUDA array to memory until it is needed, rather than doing it right away, which can improve performance in certain cases
		printf("Ipc Event Supported: %d\n", prop.ipcEventSupported);//It tells you if the GPU can use events for synchronization between different processes or across multiple GPUs.
		printf("Cluster Launch: %d\n", prop.clusterLaunch);//It tells you if the GPU can coordinate and launch tasks that span multiple GPU clusters, which can be useful for large-scale, multi-GPU workloads.
		printf("Unified Function Pointers: %d\n", prop.unifiedFunctionPointers);//It tells you if you can use the same function pointers in both CPU and GPU code, simplifying the management of function calls across the host and device.
	}

		printf("\n");
	return(0);
}


