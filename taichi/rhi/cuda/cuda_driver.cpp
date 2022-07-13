#include "taichi/rhi/cuda/cuda_driver.h"

#include "taichi/system/dynamic_loader.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/util/environ_config.h"

#include "taichi/debug/log.h"

TLANG_NAMESPACE_BEGIN

std::string get_cuda_error_message(uint32 err) {
  auto err_name_ptr = CUDADriver::get_instance_without_context().get_error_name(err);
  auto err_string_ptr = CUDADriver::get_instance_without_context().get_error_string(err);
  auto a = fmt::format("CUDA Error {}: {}", err_name_ptr, err_string_ptr);
  std::cout << "Gale | " << " a = " << a << std::endl;
  err_name_ptr = CUDADriver::get_instance_without_context().get_error_name(1);
  err_string_ptr = CUDADriver::get_instance_without_context().get_error_string(1);
  auto b = fmt::format("CUDA Error {}: {}", err_name_ptr, err_string_ptr);
  std::cout << "Gale | " << " b = " << b << std::endl;

  int version = 1;
  CUDADriver::get_instance_without_context().driver_get_version(&version);
  std::cout << "Gale | " << " version = " << version << std::endl;
  return a;
}

bool CUDADriver::detected() {
  return !disabled_by_env_ && cuda_version_valid_ && loader_->loaded();
}

CUDADriver::CUDADriver() {
  disabled_by_env_ = (get_environ_config("TI_ENABLE_CUDA", 1) == 0);
  if (disabled_by_env_) {
    TI_TRACE(
        "CUDA driver disabled by environment variable \"TI_ENABLE_CUDA\".");
    return;
  }

#if defined(TI_PLATFORM_LINUX)
  loader_ = std::make_unique<DynamicLoader>("libamdhip64.so");
#elif defined(TI_PLATFORM_WINDOWS)
  loader_ = std::make_unique<DynamicLoader>("nvcuda.dll");
#else
  static_assert(false, "Taichi CUDA driver supports only Windows and Linux.");
#endif

  if (!loader_->loaded()) {
    TI_WARN("CUDA driver not found.");
    return;
  }
  tick;
  loader_->load_function("hipGetErrorName", get_error_name);
  loader_->load_function("hipGetErrorString", get_error_string);
  loader_->load_function("hipDriverGetVersion", driver_get_version);

  int version;
  driver_get_version(&version);
  TI_TRACE("CUDA driver API (v{}.{}) loaded.", version / 1000,
           version % 1000 / 10);

  cuda_version_valid_ = true;
#define PER_CUDA_FUNCTION(name, symbol_name, ...) \
  name.set(loader_->load_function(#symbol_name)); \
  name.set_lock(&lock_);                          \
  name.set_names(#name, #symbol_name);
#include "taichi/rhi/cuda/cuda_driver_functions.inc.h"
#undef PER_CUDA_FUNCTION
// TODO
tick;
}

// This is for initializing the CUDA driver itself
CUDADriver &CUDADriver::get_instance_without_context() {
  // Thread safety guaranteed by C++ compiler
  // Note this is never deleted until the process finishes
  static CUDADriver *instance = new CUDADriver();
  return *instance;
}

CUDADriver &CUDADriver::get_instance() {
  // initialize the CUDA context so that the driver APIs can be called later
  CUDAContext::get_instance();
  return get_instance_without_context();
}

TLANG_NAMESPACE_END
