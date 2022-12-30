#pragma once

#include <memory>
#include <functional>

#include "taichi/inc/constants.h"
#include "taichi/util/lang_util.h"
#include "taichi/program/kernel_profiler.h"

TLANG_NAMESPACE_BEGIN

// A architecture-specific JIT module that initializes with an **LLVM** module
// and allows the user to call its functions
// TODO: should we generalize this to include the Metal and OpenGL backends as
// well?

class JITModule {
 public:
  JITModule() {
  }

  // Lookup a serial function.
  // For example, a CPU function, or a serial GPU function
  // This function returns a function pointer
  virtual void *lookup_function(const std::string &name) = 0;

  // Unfortunately, this can't be virtual since it's a template function
  template <typename... Args>
  std::function<void(Args...)> get_function(const std::string &name) {
    using FuncT = typename std::function<void(Args...)>;
    auto ret = FuncT((function_pointer_type<FuncT>)lookup_function(name));
    TI_ASSERT(ret != nullptr);
    return ret;
  }

  static int get_args_bytes() {
    return 0;
  }

  template <typename... Args, typename T>
  static int get_args_bytes(T t, Args ...args) {
    return get_args_bytes(args...) + sizeof(T); 
  }

  static void init_args_pointers(char *packed_args) {
    return ;
  }

  template <typename... Args, typename T>
  static void init_args_pointers(char *packed_args, T t, Args ...args) {
      std::memcpy(packed_args, &t, sizeof(t));
      init_args_pointers(packed_args + sizeof(t), args...);
      return ;
  }

  static std::vector<void *> get_arg_pointers() {
    return std::vector<void *>();
  }

  template <typename... Args, typename T>
  static std::vector<void *> get_arg_pointers(T &t, Args &...args) {
    auto ret = get_arg_pointers(args...);
    ret.insert(ret.begin(), &t);
    return ret;
  }

#if defined(TI_WITH_AMDGPU)
  template <typename... Args>
  void call(const std::string &name, Args... args) {
    if (direct_dispatch()) {
      get_function<Args...>(name)(args...);
    } else {
      auto arg_bytes = JITModule::get_args_bytes(args...);
      char *packed_args = (char*)std::malloc(arg_bytes);
      *(int *)packed_args = (int)arg_bytes;
      JITModule::init_args_pointers(packed_args, args...);
      call(name, { (void*)packed_args , (void*)&arg_bytes});
      std::free(packed_args);
    }
  }
#else

  // Note: **call** is for serial functions
  // Note: args must pass by value
  template <typename... Args>
  void call(const std::string &name, Args... args) {
    if (direct_dispatch()) {
      get_function<Args...>(name)(args...);
    } else {
      auto arg_pointers = JITModule::get_arg_pointers(args...);
      call(name, arg_pointers);
    }
  }
#endif

  virtual void call(const std::string &name,
                    const std::vector<void *> &arg_pointers) {
    TI_NOT_IMPLEMENTED
  }

  virtual void call(const std::string &name,
                    void *arg_pointers,
                    int ) {
    TI_NOT_IMPLEMENTED
  }

  // Note: **launch** is for parallel (GPU)_kernels
  // Note: args must pass by value
  template <typename... Args>
  void launch(const std::string &name,
              std::size_t grid_dim,
              std::size_t block_dim,
              std::size_t shared_mem_bytes,
              Args... args) {
    auto arg_pointers = JITModule::get_arg_pointers(args...);
    launch(name, grid_dim, block_dim, shared_mem_bytes,
                             arg_pointers);
  }

  virtual void launch(const std::string &name,
                      std::size_t grid_dim,
                      std::size_t block_dim,
                      std::size_t shared_mem_bytes,
                      const std::vector<void *> &arg_pointers) {
    TI_NOT_IMPLEMENTED
  }


  // directly call the function (e.g. on CPU), or via another runtime system
  // (e.g. cudaLaunch)?
  virtual bool direct_dispatch() const = 0;

  virtual ~JITModule() {
  }
};

TLANG_NAMESPACE_END
