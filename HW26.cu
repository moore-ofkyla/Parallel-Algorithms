for (int i = 0; i < NumberOfGpus; i++) {
    int offset = i * NPerGPU;
    int workload = min(NPerGPU, N - offset);

    cudaMemPrefetchAsync(&P[offset], workload * sizeof(float3), i);
    cudaMemPrefetchAsync(&V[offset], workload * sizeof(float3), i);
    cudaMemPrefetchAsync(&F[offset], workload * sizeof(float3), i);
    cudaMemPrefetchAsync(&M[offset], workload * sizeof(float), i);

    cudaSetDevice(i);
    getForces<<<GridSize, BlockSize>>>(P, V, F, M, G, H, NPerGPU, N, i);
    cudaErrorCheck(__FILE__, __LINE__);
    moveBodies<<<GridSize, BlockSize>>>(P, V, F, M, Damp, dt, t, NPerGPU, N, i);
    cudaErrorCheck(__FILE__, __LINE__);
}
