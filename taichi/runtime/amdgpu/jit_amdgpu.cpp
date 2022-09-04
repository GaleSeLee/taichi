#include "taichi/runtime/amdgpu/jit_amdgpu.h"
#include "taichi/runtime/llvm/llvm_context.h"

TLANG_NAMESPACE_BEGIN

#if defined(TI_WITH_AMDGPU)

std::string load_hsaco(const std::string& filename) {
  std::ifstream src_file(filename);
  if (!src_file.is_open()) {
      TI_ERROR(fmt::format("Open {} Error", filename));
  }
  return std::string(std::istreambuf_iterator<char>(src_file), (std::istreambuf_iterator<char>()));
}

JITModule *JITSessionAMDGPU ::add_module(std::unique_ptr<llvm::Module> M,
                                       int max_reg) {
  auto gcn = compile_module_to_gcn(M);

  static FileSequenceWriter writer("taichi_kernel_gcn_{:04d}.o",
                                     "module AMDGCN");
  writer.write(gcn);

  auto filename = writer.get_filename();
  auto obj_filename = filename + ".o";
  auto hsaco_filename = filename + ".hsaco";

  // TODO: figure out why using the guard leads to wrong tests results
  // auto context_guard = AMDGPUContext::get_instance().get_guard();
  AMDGPUContext::get_instance().make_current();
  // Create module for object
  void *amdgpu_module;
  TI_TRACE("PTX size: {:.2f}KB", ptx.size() / 1024.0);
  auto t = Time::get_time();
  TI_TRACE("Loading module...");
  [[maybe_unused]] auto _ = AMDGPUContext::get_instance().get_lock_guard();

  std::string lld_cmd = "ld.lld -shared " + obj_filename + " -o " + hsaco_filename;

  if (std::system(lld_cmd.c_str())) 
      TI_ERROR(fmt::format("Generate {} Error", hsaco_filename));

  std::string hsaoc_str = load_file(hsaco_filename);
  AMDGPUDriver::get_instance().module_load_data(&amdgpu_module, hsaco_str.c_str());
  TI_TRACE("AMDGPU module load time : {}ms", (Time::get_time() - t) * 1000);
  modules.push_back(std::make_unique<JITModuleAMDGPU>(amdgpu_module));
  return modules.back().get();
}

std::string JITSessionCUDA::compile_module_to_gcn(
  // Note: compile_module_to_gcn generates bianry code object actually.
  std::unique_ptr<llvm::Module> &llvm_module) {
  // Part of this function is borrowed from Halide::CodeGen_PTX_Dev.cpp
  if (llvm::verifyModule(*llvm_module, &llvm::errs())) {
    llvm_module->print(llvm::errs(), nullptr);
    TI_WARN("Module broken");
  }

  using namespace llvm;

  if (this->config_->print_kernel_llvm_ir) {
    static FileSequenceWriter writer("taichi_kernel_amdgpu_llvm_ir_{:04d}.ll",
                                     "unoptimized LLVM IR (AMDGPU)");
    writer.write(llvm_module.get());
  }
  auto triple_str = llvm_module->getTargetTriple(); 
  std::string error_str;
  auto target = llvm::TargetRegistry::lookupTarget(triple_str, error_str);
  llvm::TargetOptions options;
  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      "amdgcn-amd-amdhsa", "gfx1030", "", options, llvm::Reloc::PIC_, 
      llvm::CodeModel::Small, llvm::CodeGenOpt::Aggressive));
    
  llvm_module->setDataLayout(machine->createDataLayout());

  llvm::legacy::FunctionPassManager function_pass_manager(llvm_module.get());
  llvm::legacy::PassManager module_pass_manager;

  module_pass_manager.add(llvm::createTargetTransformInfoWrapperPass(machine->getTargetIRAnalysis()));
  function_pass_manager.add(llvm::createTargetTransformInfoWrapperPass(machine->getTargetIRAnalysis()));

  llvm::PassManagerBuilder builder;
  builder.OptLevel = 3;
  builder.Inliner = llvm::createFunctionInliningPass(builder.OptLevel, 0, false);
  machine->adjustPassManager(builder);
  builder.populateFunctionPassManager(function_pass_manager);
  builder.populateModulePassManager(module_pass_manager);

  machine->Options.MCOptions.AsmVerbose = true;

  llvm::SmallString<0> outstr;
  llvm::raw_svector_ostream llvm_stream(outstr);

  machine->addPassesToEmitFile(module_pass_manager, llvm_stream, nullptr, llvm::CGFT_ObjectFile, true);
  function_pass_manager.doInitialization();
  for (auto func = llvm_module->begin(); func != llvm_module->end(); ++func)
    function_pass_manager.run(*func);
  function_pass_manager.doFinalization();
  module_pass_manager.run(*llvm_module);

  std::string obj_str(outstr.begin(), outstr.end());
  return obj_str;
}

std::unique_ptr<JITSession> create_llvm_jit_session_amdgpu(
    TaichiLLVMContext *tlctx,
    CompileConfig *config,
    Arch arch) {
  TI_ASSERT(arch == Arch::amdgpu);
  // TODO (Gale)
  auto data_layout = llvm::DataLayout(
      "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-"
      "f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64");
  return std::make_unique<JITSessionAMDGPU>(tlctx, config, data_layout);
}
#else
std::unique_ptr<JITSession> create_llvm_jit_session_amdgpu(
    TaichiLLVMContext *tlctx,
    CompileConfig *config,
    Arch arch) {
  TI_NOT_IMPLEMENTED
}
#endif

TLANG_NAMESPACE_END