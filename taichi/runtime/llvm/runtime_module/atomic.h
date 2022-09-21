#pragma once

// These instructions should be replaced by CUDA intrinsics on GPUs
#ifdef ARCH_amdgpu
#include <hip/hip_runtime.h>
#endif

#ifdef ARCH_amdgpu
#define DEFINE_ATOMIC_EXCHANGE(T)                               \
  __host__ __device__                                           \
  T atomic_exchange_##T(volatile T *dest, T val) {              \
    T ret;                                                      \
    __atomic_exchange(dest, &val, &ret,                         \
                      std::memory_order::memory_order_seq_cst); \
    return ret;                                                 \
  }
#else
#define DEFINE_ATOMIC_EXCHANGE(T)                               \
  T atomic_exchange_##T(volatile T *dest, T val) {              \
    T ret;                                                      \
    __atomic_exchange(dest, &val, &ret,                         \
                      std::memory_order::memory_order_seq_cst); \
    return ret;                                                 \
  }
#endif

DEFINE_ATOMIC_EXCHANGE(i32)
DEFINE_ATOMIC_EXCHANGE(i64)
DEFINE_ATOMIC_EXCHANGE(u32)
DEFINE_ATOMIC_EXCHANGE(u64)

#ifdef ARCH_amdgpu
#define DEFINE_ATOMIC_OP_INTRINSIC(OP, T)                                \
  __host__ __device__                                                    \
  T atomic_##OP##_##T(volatile T *dest, T val) {                         \
    return __atomic_fetch_##OP(dest, val,                                \
                               std::memory_order::memory_order_seq_cst); \
  }
#else
#define DEFINE_ATOMIC_OP_INTRINSIC(OP, T)                                \
  T atomic_##OP##_##T(volatile T *dest, T val) {                         \
    return __atomic_fetch_##OP(dest, val,                                \
                               std::memory_order::memory_order_seq_cst); \
  }
#endif


DEFINE_ATOMIC_OP_INTRINSIC(add, i32)
DEFINE_ATOMIC_OP_INTRINSIC(add, i64)
DEFINE_ATOMIC_OP_INTRINSIC(and, i32)
DEFINE_ATOMIC_OP_INTRINSIC(and, i64)
DEFINE_ATOMIC_OP_INTRINSIC(and, u32)
DEFINE_ATOMIC_OP_INTRINSIC(and, u64)
DEFINE_ATOMIC_OP_INTRINSIC(or, i32)
DEFINE_ATOMIC_OP_INTRINSIC(or, i64)
DEFINE_ATOMIC_OP_INTRINSIC(or, u32)
DEFINE_ATOMIC_OP_INTRINSIC(or, u64)
DEFINE_ATOMIC_OP_INTRINSIC(xor, i32)
DEFINE_ATOMIC_OP_INTRINSIC(xor, i64)
DEFINE_ATOMIC_OP_INTRINSIC(xor, u32)
DEFINE_ATOMIC_OP_INTRINSIC(xor, u64)

#ifdef ARCH_amdgpu
#define DEFINE_ADD(T)                     \
__host__ __device__ T add_##T(T a, T b) { \
    return a + b;                         \
  }
#else
#define DEFINE_ADD(T)   \
  T add_##T(T a, T b) { \
    return a + b;       \
  }
#endif

#ifdef ARCH_amdgpu
#define DEFINE_MIN(T)                       \
__host__ __device__ T min_##T(T a, T b) {   \
    return b > a ? a : b;                   \
  }
#else
#define DEFINE_MIN(T)     \
  T min_##T(T a, T b) {   \
    return b > a ? a : b; \
  }
#endif 

#ifdef ARCH_amdgpu         
#define DEFINE_MAX(T)                       \
__host__ __device__ T max_##T(T a, T b) {   \
    return b < a ? a : b;                   \
  }
#else
#define DEFINE_MAX(T)     \
  T max_##T(T a, T b) {   \
    return b < a ? a : b; \
  }
#endif

#ifdef ARCH_amdgpu                                                              
#define DEFINE_ATOMIC_OP_COMP_EXCH(OP, T)                                     \
__host__ __device__ T atomic_##OP##_##T(volatile T *dest, T inc) {            \
    T old_val;                                                                \
    T new_val;                                                                \
    do {                                                                      \
      old_val = *dest;                                                        \
      new_val = OP##_##T(old_val, inc);                                       \
    } while (                                                                 \
        !__atomic_compare_exchange(dest, &old_val, &new_val, true,            \
                                   std::memory_order::memory_order_seq_cst,   \
                                   std::memory_order::memory_order_seq_cst)); \
    return old_val;                                                           \
  }
#else
#define DEFINE_ATOMIC_OP_COMP_EXCH(OP, T)                                     \
  T atomic_##OP##_##T(volatile T *dest, T inc) {                              \
    T old_val;                                                                \
    T new_val;                                                                \
    do {                                                                      \
      old_val = *dest;                                                        \
      new_val = OP##_##T(old_val, inc);                                       \
    } while (                                                                 \
        !__atomic_compare_exchange(dest, &old_val, &new_val, true,            \
                                   std::memory_order::memory_order_seq_cst,   \
                                   std::memory_order::memory_order_seq_cst)); \
    return old_val;                                                           \
  }
#endif

DEFINE_ADD(f32)
DEFINE_ADD(f64)
DEFINE_MIN(f32)
DEFINE_MIN(f64)
DEFINE_MAX(f32)
DEFINE_MAX(f64)

DEFINE_ATOMIC_OP_COMP_EXCH(add, f32)
DEFINE_ATOMIC_OP_COMP_EXCH(add, f64)
DEFINE_ATOMIC_OP_COMP_EXCH(min, i32)
DEFINE_ATOMIC_OP_COMP_EXCH(min, i64)
DEFINE_ATOMIC_OP_COMP_EXCH(min, f32)
DEFINE_ATOMIC_OP_COMP_EXCH(min, f64)
DEFINE_ATOMIC_OP_COMP_EXCH(max, i32)
DEFINE_ATOMIC_OP_COMP_EXCH(max, i64)
DEFINE_ATOMIC_OP_COMP_EXCH(max, f32)
DEFINE_ATOMIC_OP_COMP_EXCH(max, f64)
DEFINE_ATOMIC_OP_COMP_EXCH(min, u32)
DEFINE_ATOMIC_OP_COMP_EXCH(min, u64)
DEFINE_ATOMIC_OP_COMP_EXCH(max, u32)
DEFINE_ATOMIC_OP_COMP_EXCH(max, u64)