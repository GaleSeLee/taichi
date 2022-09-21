extern "C" {
#ifdef ARCH_amdgpu
__host__ __device__ 
#endif
int thread_idx() {
  return 0;
}

#ifdef ARCH_amdgpu
__host__ __device__ 
#endif
int warp_size() {
  return 32;
}

#ifdef ARCH_amdgpu
__host__ __device__ 
#endif
int warp_idx() {
  return thread_idx() % warp_size();
}

#ifdef ARCH_amdgpu
__host__ __device__ 
#endif
int block_idx() {
  return 0;
}

#ifdef ARCH_amdgpu
__host__ __device__ 
#endif
int block_dim() {
  return 1;
}

#ifdef ARCH_amdgpu
__host__ __device__ 
#endif
int grid_dim() {
  return 1;
}
}