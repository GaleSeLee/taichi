#define TI_RUNTIME_HOST
#include "amdgpu_context.h"

#include <unordered_map>
#include <mutex>

#include "taichi/util/lang_util.h"
#include "taichi/program/program.h"
#include "taichi/system/threading.h"
#include "taichi/rhi/amdgpu/amdgpu_driver.h"
#include "taichi/analysis/offline_cache_util.h"

TLANG_NAMESPACE_BEGIN

AMDGPUContext::AMDGPUContext()
    : profiler_(nullptr), driver_(AMDGPUDriver::get_instance_without_context()) {
  dev_count_ = 0;
  driver_.init(0);
  driver_.device_get_count(&dev_count_);
  driver_.device_get(&device_, 0);

  char name[128];
  driver_.device_get_name(name, 128, device_);

  TI_TRACE("Using AMDGPU device [id=0]: {}", name);

  // TODO: Find the way to get the arch version (gfx1030)

  // int cc_major, cc_minor;
  // driver_.device_get_attribute(
  //     &cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0);
  // driver_.device_get_attribute(
  //     &cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 0);

  driver_.context_create(&context_, 0, device_);

  const auto GB = std::pow(1024.0, 3.0);
  TI_TRACE("Total memory {:.2f} GB; free memory {:.2f} GB",
           get_total_memory() / GB, get_free_memory() / GB);

  // compute_capability_ = cc_major * 10 + cc_minor;

  // if (compute_capability_ > 75) {
    // The NVPTX backend of LLVM 10.0.0 does not seem to support
    // compute_capability > 75 yet. See
    // llvm-10.0.0.src/build/lib/Target/NVPTX/NVPTXGenSubtargetInfo.inc
  //   compute_capability_ = 75;
  // }

  // mcpu_ = fmt::format("sm_{}", compute_capability_);
  mcpu_ = "gfx1030";

  TI_TRACE("Emitting AMDGPU code for {}", mcpu_);
}

std::size_t AMDGPUContext::get_total_memory() {
  std::size_t ret, _;
  driver_.mem_get_info(&_, &ret);
  return ret;
}

std::size_t AMDGPUContext::get_free_memory() {
  std::size_t ret, _;
  driver_.mem_get_info(&ret, &_);
  return ret;
}

std::string AMDGPUContext::get_device_name() {
  constexpr uint32_t kMaxNameStringLength = 128;
  char name[kMaxNameStringLength];
  driver_.device_get_name(name, kMaxNameStringLength /*=128*/, device_);
  std::string str(name);
  return str;
}

void AMDGPUContext::launch(void *func,
                         const std::string &task_name,
                         void *arg_pointers,
                         unsigned grid_dim,
                         unsigned block_dim,
                         std::size_t dynamic_shared_mem_bytes,
                         int arg_bytes) {
  KernelProfilerBase::TaskHandle task_handle;
  if (grid_dim > 0) {
    std::lock_guard<std::mutex> _(lock_);
    void *config[] = {(void *)0x01, const_cast<void*>(arg_pointers), 
                      (void *)0x02, &arg_bytes, (void *)0x03}; 
    driver_.launch_kernel(func, grid_dim, 1, 1, block_dim, 1, 1,
                          dynamic_shared_mem_bytes, nullptr,
                          nullptr, reinterpret_cast<void**> (&config));
  }
  if (debug_) {
    driver_.stream_synchronize(nullptr);
  }
}

AMDGPUContext::~AMDGPUContext() {
}

AMDGPUContext &AMDGPUContext::get_instance() {
  static auto context = new AMDGPUContext();
  return *context;
}

TLANG_NAMESPACE_END