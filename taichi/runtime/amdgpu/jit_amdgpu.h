#include <memory>
#include <utility>
#include <mutex>
#include <random>
#include <unistd.h>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#ifdef TI_LLVM_15
#include "llvm/MC/TargetRegistry.h"
#else
#include "llvm/Support/TargetRegistry.h"
#endif
#include "llvm/Target/TargetMachine.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"

#include "taichi/rhi/amdgpu/amdgpu_context.h"
#include "taichi/rhi/amdgpu/amdgpu_driver.h"
#include "taichi/jit/jit_session.h"
#include "taichi/util/lang_util.h"
#include "taichi/program/program.h"
#include "taichi/system/timer.h"
#include "taichi/util/file_sequence_writer.h"
#include "taichi/util/io.h"

#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

TLANG_NAMESPACE_BEGIN

#if defined(TI_WITH_AMDGPU)

uint64 get_random_num() {
  static std::random_device device("/dev/urandom");
  static std::mt19937_64* rng = new std::mt19937_64(device());
  return (*rng)();
}

class JITModuleAMDGPU : public JITModule {
 private:
  void *module_;

 public:
  explicit JITModuleAMDGPU(void *module) : module_(module) {
  }

  void *lookup_function(const std::string &name) override {
    // TODO: figure out why using the guard leads to wrong tests results
    // auto context_guard = AMDGPUContext::get_instance().get_guard();
    AMDGPUContext::get_instance().make_current();
    void *func = nullptr;
    auto t = Time::get_time();
    auto err = AMDGPUDriver::get_instance().module_get_function.call_with_warning(
        &func, module_, name.c_str());
    if (err) {
      TI_ERROR("Cannot look up function {}", name);
    }
    t = Time::get_time() - t;
    TI_TRACE("AMDGPU module_get_function {} costs {} ms", name, t * 1000);
    TI_ASSERT(func != nullptr);
    return func;
  }

  void call(const std::string &name,
            void* arg_pointers,
            int arg_bytes) override {
    launch(name, 1, 1, 0, arg_pointers, arg_bytes);
  }

  void launch(const std::string &name,
              std::size_t grid_dim,
              std::size_t block_dim,
              std::size_t dynamic_shared_mem_bytes,
              void *arg_pointers,
              int arg_bytes) override {
    auto func = lookup_function(name);
    AMDGPUContext::get_instance().launch(func, name, arg_pointers,
                                         grid_dim, block_dim,
                                         dynamic_shared_mem_bytes, arg_bytes);
  }

  bool direct_dispatch() const override {
    return false;
  }

};

class JITSessionAMDGPU : public JITSession {
 public:
  llvm::DataLayout data_layout;

  JITSessionAMDGPU(TaichiLLVMContext *tlctx,
                 CompileConfig *config,
                 llvm::DataLayout data_layout)
      : JITSession(tlctx, config), data_layout(data_layout) {
        random_num_ = get_random_num();
  }

  JITModule *add_module(std::unique_ptr<llvm::Module> M, int max_reg) override;

  llvm::DataLayout get_data_layout() override {
    return data_layout;
  }

  std::string load_hsaco(const std::string& filename) {
    std::ifstream src_file(filename);
    if (!src_file.is_open()) {
        TI_ERROR(fmt::format("Open {} Error", filename));
    }
    return std::string(std::istreambuf_iterator<char>(src_file), (std::istreambuf_iterator<char>()));
  }

  std::string get_tmp_dir() {
    char *env_dir = std::getenv("TI_TMP_DIR");
    std::string tmp_dir = "/tmp/taichi_hsaco/";
    if (env_dir) {
      tmp_dir = env_dir;
      if (tmp_dir[tmp_dir.size() - 1] != '/') {
        tmp_dir += '/';
      }
    }
    tmp_dir += std::to_string(random_num_) + "/";
    create_directories(tmp_dir);
    return tmp_dir;
  }

 private:
  std::string compile_module_to_hsaco(std::unique_ptr<llvm::Module> &module);
  uint64_t random_num_;
};

#endif

std::unique_ptr<JITSession> create_llvm_jit_session_amdgpu(
    TaichiLLVMContext *tlctx,
    CompileConfig *config,
    Arch arch);

TLANG_NAMESPACE_END