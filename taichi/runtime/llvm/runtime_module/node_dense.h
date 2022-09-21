#pragma once

// Specialized Attributes and functions
struct DenseMeta : public StructMeta {
  int morton_dim;
};

STRUCT_FIELD(DenseMeta, morton_dim)

#ifdef ARCH_amdgpu
#include <hip/hip_runtime.h>
#endif

#ifdef ARCH_amdgpu
__host__ __device__
#endif
i32 Dense_get_num_elements(Ptr meta, Ptr node) {
  return ((StructMeta *)meta)->max_num_elements;
}

#ifdef ARCH_amdgpu
__host__ __device__
#endif
void Dense_activate(Ptr meta, Ptr node, int i) {
  // Dense elements are always active
}

#ifdef ARCH_amdgpu
__host__ __device__
#endif
i32 Dense_is_active(Ptr meta, Ptr node, int i) {
  return 1;
}

#ifdef ARCH_amdgpu
__host__ __device__
#endif
Ptr Dense_lookup_element(Ptr meta, Ptr node, int i) {
  return node + ((StructMeta *)meta)->element_size * i;
}