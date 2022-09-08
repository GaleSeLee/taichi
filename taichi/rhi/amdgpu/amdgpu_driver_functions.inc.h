// Init
PER_CUDA_FUNCTION(init, hipInit, unsigned int);

// Device management
PER_CUDA_FUNCTION(device_get_count, hipGetDeviceCount, int *);
PER_CUDA_FUNCTION(device_get_attribute, hipDeviceGetAttribute, int *, uint32, int); 
PER_CUDA_FUNCTION(device_get_name, hipDeviceGetName, char *, int, void *);
PER_CUDA_FUNCTION(device_get, hipDeviceGet, void *, void *);

// Context management
PER_CUDA_FUNCTION(context_create, hipCtxCreate, void*, int, void *);
PER_CUDA_FUNCTION(context_set_current, hipCtxSetCurrent, void *);
PER_CUDA_FUNCTION(context_get_current, hipCtxGetCurrent, void **);

// Stream management
PER_CUDA_FUNCTION(stream_create, hipStreamCreate, void **, uint32);

// Memory management
PER_CUDA_FUNCTION(memcpy_host_to_device, hipMemcpyHtoD, void *, void *, std::size_t);
PER_CUDA_FUNCTION(memcpy_device_to_host, hipMemcpyDtoH, void *, void *, std::size_t);
PER_CUDA_FUNCTION(memcpy_device_to_device, hipMemcpyDtoD, void *, void *, std::size_t);
PER_CUDA_FUNCTION(memcpy, hipMemcpy, void *, void *, std::size_t, unsigned int);
PER_CUDA_FUNCTION(memcpy_async, hipMemcpyAsync, void *, void *, std::size_t, unsigned int, void *);
PER_CUDA_FUNCTION(memcpy_host_to_device_async, hipMemcpyHtoDAsync, void *, void *, std::size_t, void *);
PER_CUDA_FUNCTION(memcpy_device_to_host_async, hipMemcpyDtoHAsync, void *, void *, std::size_t, void *);
PER_CUDA_FUNCTION(malloc, hipMalloc, void **, std::size_t);
PER_CUDA_FUNCTION(malloc_managed, hipMallocManaged, void **, std::size_t, uint32);
PER_CUDA_FUNCTION(memset, hipMemset, void *, uint8, std::size_t);
PER_CUDA_FUNCTION(mem_free, hipFree, void *);
PER_CUDA_FUNCTION(mem_get_info, hipMemGetInfo, std::size_t *, std::size_t *);
PER_CUDA_FUNCTION(mem_get_attribute, hipPointerGetAttribute, void *, uint32, void *);

// Module and kernels
PER_CUDA_FUNCTION(module_get_function, hipModuleGetFunction, void **, void *, const char *);
PER_CUDA_FUNCTION(module_load_data_ex, hipModuleLoadDataEx, void **, const char *,
                    uint32, uint32 *, void **);
PER_CUDA_FUNCTION(module_load_data, hipModuleLoadData, void **, const void *);
PER_CUDA_FUNCTION(launch_kernel, hipModuleLaunchKernel, void *, uint32, uint32, uint32,
                    uint32, uint32, uint32, uint32, void *, void **, void **);
PER_CUDA_FUNCTION(kernel_get_attribute, hipFuncGetAttribute, int *, uint32, void *);
PER_CUDA_FUNCTION(kernel_get_occupancy, hipOccupancyMaxActiveBlocksPerMultiprocessor,
                    int *, void *, int, size_t);

// Stream management
PER_CUDA_FUNCTION(stream_synchronize, hipStreamSynchronize, void *);

// Event management
PER_CUDA_FUNCTION(event_create, hipEventCreateWithFlags, void **, uint32);
PER_CUDA_FUNCTION(event_destroy, hipEventDestroy, void *);
PER_CUDA_FUNCTION(event_record, hipEventRecord, void *, void *);
PER_CUDA_FUNCTION(event_synchronize, hipEventSynchronize, void *);
PER_CUDA_FUNCTION(event_elapsed_time, hipEventElapsedTime, float *, void *, void *);