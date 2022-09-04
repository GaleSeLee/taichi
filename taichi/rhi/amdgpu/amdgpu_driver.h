#pragma once

#include <mutex>

#include "taichi/system/dynamic_loader.h"

TLANG_NAMESPACE_BEGIN

// Driver constants from cuda.h

constexpr uint32 HIP_EVENT_DEFAULT = 0x0;
constexpr uint32 HIP_STREAM_DEFAULT = 0x0;
constexpr uint32 HIP_STREAM_NON_BLOCKING = 0x1;
constexpr uint32 HIP_MEM_ATTACH_GLOBAL = 0x1;
constexpr uint32 HIP_MEM_ADVISE_SET_PREFERRED_LOCATION = 3;
constexpr uint32 HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 26;
constexpr uint32 HIP_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 25;
constexpr uint32 HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 63;
// The Capability Major and minor don't represent the version of ROCm
// Currently, I don't know the meaning of these
constexpr uint32 HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 23;
constexpr uint32 HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 61;
constexpr uint32 HIP_ERROR_ASSERT = 710;
constexpr uint32 HIP_JIT_MAX_REGISTERS = 0;
constexpr uint32 HIP_POINTER_ATTRIBUTE_MEMORY_TYPE = 2;
constexpr uint32 HIP_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41;
constexpr uint32 HIP_SUCCESS = 0;
constexpr uint32 HIP_MEMORYTYPE_DEVICE = 2;

std::string get_amdgpu_error_message(uint32 err);

template <typename... Args>
class AMDGPUFunction {
 public:
  AMDGPUFunction() {
    function_ = nullptr;
  }

  void set(void *func_ptr) {
    function_ = (func_type *)func_ptr;
  }

  uint32 call(Args... args) {
    TI_ASSERT(function_ != nullptr);
    TI_ASSERT(driver_lock_ != nullptr);
    std::lock_guard<std::mutex> _(*driver_lock_);
    return (uint32)function_(args...);
  }

  void set_names(const std::string &name, const std::string &symbol_name) {
    name_ = name;
    symbol_name_ = symbol_name;
  }

  void set_lock(std::mutex *lock) {
    driver_lock_ = lock;
  }

  std::string get_error_message(uint32 err) {
    return get_cuda_error_message(err) +
           fmt::format(" while calling {} ({})", name_, symbol_name_);
  }

  uint32 call_with_warning(Args... args) {
    auto err = call(args...);
    TI_WARN_IF(err, "{}", get_error_message(err));
    return err;
  }

  // Note: CUDA driver API passes everything as value
  void operator()(Args... args) {
    auto err = call(args...);
    TI_ERROR_IF(err, get_error_message(err));
  }

 private:
  using func_type = uint32_t(Args...);

  func_type *function_{nullptr};
  std::string name_, symbol_name_;
  std::mutex *driver_lock_{nullptr};
};

class AMDGPUDriverBase {
 public:
  ~AMDGPUDriverBase() = default;

 protected:
  std::unique_ptr<DynamicLoader> loader_;
  AMDGPUDriverBase();

  bool load_lib(std::string lib_linux);

  bool disabled_by_env_{false};
};

class AMDGPUDriver : protected AMDGPUDriverBase {
 public:
 // Actually, functions are runtime api
 // This is not the same as CUDA
 // The name "amdgpu_driver_functions.inc.h" is to unify with CUDA
#define PER_AMDGPU_FUNCTION(name, symbol_name, ...) \
  AMDGPUFunction<__VA_ARGS__> name;
#include "taichi/rhi/amdgpu/amdgpu_driver_functions.inc.h" 
#undef PER_AMDGPU_FUNCTION

  char (*get_error_name)(uint32);

  char (*get_error_string)(uint32);

  void (*driver_get_version)(int *);

  bool detected();

  static CUDADriver &get_instance();

  static CUDADriver &get_instance_without_context();

 private:
  CUDADriver();

  std::mutex lock_;

  // bool rocm_version_valid_{false};
};

TLANG_NAMESPACE_END