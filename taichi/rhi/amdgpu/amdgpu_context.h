#pragma once

#include <mutex>
#include <unordered_map>
#include <thread>

#include "taichi/program/kernel_profiler.h"
#include "taichi/rhi/amdgpu/amdgpu_driver.h"

TLANG_NAMESPACE_BEGIN

class AMDGPUDriver;

class AMDGPUContext {
 private:
  void *device_;
  void *context_;
  int dev_count_;
  std::string mcpu_;
  std::mutex lock_;
  AMDGPUDriver &driver_;
  bool debug_;

 public:
  AMDGPUContext();

  std::size_t get_total_memory();
  std::size_t get_free_memory();
  std::string get_device_name();

  bool detected() const {
    return dev_count_ != 0;
  }

  void launch(void *func,
              const std::string &task_name,
              void* arg_pointers,
              unsigned grid_dim,
              unsigned block_dim,
              std::size_t dynamic_shared_mem_bytes,
              int arg_bytes);

  void set_debug(bool debug) {
    debug_ = debug;
  }

  std::string get_mcpu() const {
    return mcpu_;
  }

  void *get_context() {
    return context_;
  }

  void make_current() {
    driver_.context_set_current(context_);
  }

  ~CUDAContext();

  class ContextGuard {
   private:
    void *old_ctx_;
    void *new_ctx_;

   public:
    ContextGuard(CUDAContext *new_ctx) : old_ctx_(nullptr), new_ctx_(new_ctx) {
      CUDADriver::get_instance().context_get_current(&old_ctx_);
      if (old_ctx_ != new_ctx)
        new_ctx->make_current();
    }

    ~ContextGuard() {
      if (old_ctx_ != new_ctx_) {
        CUDADriver::get_instance().context_set_current(old_ctx_);
      }
    }
  };

  ContextGuard get_guard() {
    return ContextGuard(this);
  }

  std::unique_lock<std::mutex> get_lock_guard() {
    return std::unique_lock<std::mutex>(lock_);
  }

  static CUDAContext &get_instance();
};

TLANG_NAMESPACE_END