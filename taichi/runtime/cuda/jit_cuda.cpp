#include "taichi/runtime/cuda/jit_cuda.h"
#include "taichi/runtime/llvm/llvm_context.h"
#include "taichi/debug/log.h"

TLANG_NAMESPACE_BEGIN

#if defined(TI_WITH_CUDA)

JITModule *JITSessionCUDA ::add_module(std::unique_ptr<llvm::Module> M,
                                       int max_reg) {
  auto ptx = compile_module_to_ptx(M);
  if (this->config_->print_kernel_nvptx) {
    static FileSequenceWriter writer("taichi_kernel_amdgcn_{:04d}.o",
                                     "module NVPTX");
    writer.write(ptx);
  }
  tick;
  // TODO: figure out why using the guard leads to wrong tests results
  // auto context_guard = CUDAContext::get_instance().get_guard();
  CUDAContext::get_instance().make_current();
  // Create module for object
  void *cuda_module;
  TI_TRACE("PTX size: {:.2f}KB", ptx.size() / 1024.0);
  auto t = Time::get_time();
  TI_TRACE("Loading module...");
  [[maybe_unused]] auto _ = CUDAContext::get_instance().get_lock_guard();

  constexpr int max_num_options = 8;
  int num_options = 0;
  uint32 options[max_num_options];
  void *option_values[max_num_options];

  // Insert options
  if (max_reg != 0) {
    options[num_options] = CU_JIT_MAX_REGISTERS;
    option_values[num_options] = &max_reg;
    num_options++;
  }

  TI_ASSERT(num_options <= max_num_options);


  CUDADriver::get_instance().module_load_data(&cuda_module, (const void*)ptx.c_str());

  TI_TRACE("CUDA module load time : {}ms", (Time::get_time() - t) * 1000);
  // cudaModules.push_back(cudaModule);
  modules.push_back(std::make_unique<JITModuleCUDA>(cuda_module));
  return modules.back().get();
}

std::string cuda_mattrs() {
  return "";
}

std::string convert(std::string new_name) {
  // Evil C++ mangling on Windows will lead to "unsupported characters in
  // symbol" error in LLVM PTX printer. Convert here.
  for (int i = 0; i < (int)new_name.size(); i++) {
    if (new_name[i] == '@')
      new_name.replace(i, 1, "_at_");
    if (new_name[i] == '?')
      new_name.replace(i, 1, "_qm_");
    if (new_name[i] == '$')
      new_name.replace(i, 1, "_dl_");
    if (new_name[i] == '<')
      new_name.replace(i, 1, "_lb_");
    if (new_name[i] == '>')
      new_name.replace(i, 1, "_rb_");
    TI_ASSERT(std::isalpha(new_name[i]) || std::isdigit(new_name[i]) ||
              new_name[i] == '_' || new_name[i] == '.');
  }
  if (!new_name.empty())
    TI_ASSERT(isalpha(new_name[0]) || new_name[0] == '_' || new_name[0] == '.');
  return new_name;
}

// TODO Gale
std::string JITSessionCUDA::compile_module_to_ptx(
    std::unique_ptr<llvm::Module> &module) {
  TI_AUTO_PROF
  // Part of this function is borrowed from Halide::CodeGen_PTX_Dev.cpp
  if (llvm::verifyModule(*module, &llvm::errs())) {
    module->print(llvm::errs(), nullptr);
    TI_WARN("Module broken");
  }

  using namespace llvm;

  if (this->config_->print_kernel_llvm_ir) {
    static FileSequenceWriter writer("taichi_kernel_amdgpu_llvm_ir_{:04d}.ll",
                                     "unoptimized LLVM IR (AMDGPU)");
    writer.write(module.get());
  }
  tick;

  for (auto &f : module->globals())
    f.setName(convert(f.getName()));
  for (auto &f : *module)
    f.setName(convert(f.getName()));

  llvm::Triple triple(module->getTargetTriple());
  tickv(triple.str());

  // Allocate target machine

  std::string err_str;
  const llvm::Target *target =
      TargetRegistry::lookupTarget(triple.str(), err_str);
  TI_ERROR_UNLESS(target, err_str);

  TargetOptions options;
  /*options.PrintMachineCode = 0;
  if (this->config_->fast_math) {
    options.AllowFPOpFusion = FPOpFusion::Fast;
    // See NVPTXISelLowering.cpp
    // Setting UnsafeFPMath true will result in approximations such as
    // sqrt.approx in PTX for both f32 and f64
    options.UnsafeFPMath = 1;
    options.NoInfsFPMath = 1;
    options.NoNaNsFPMath = 1;
  } else {
    options.AllowFPOpFusion = FPOpFusion::Strict;
    options.UnsafeFPMath = 0;
    options.NoInfsFPMath = 0;
    options.NoNaNsFPMath = 0;
  }
  options.HonorSignDependentRoundingFPMathOption = 0;
  options.NoZerosInBSS = 0;
  options.GuaranteedTailCallOpt = 0;
  options.StackAlignmentOverride = 0;
*/

  tickv(triple.str());
  llvm::StringRef mcpu = llvm::sys::getHostCPUName();
  tickv(mcpu.str());
  std::unique_ptr<TargetMachine> target_machine(target->createTargetMachine(
      triple.str(), "gfx1030", "",
      options, llvm::Reloc::PIC_, llvm::CodeModel::Small,
      CodeGenOpt::Aggressive));

  TI_ERROR_UNLESS(target_machine.get(), "Could not allocate target machine!");

  tick;
  
  tickv(target_machine->createDataLayout().getStringRepresentation());
  module->setDataLayout(target_machine->createDataLayout());

  // Set up passes
  llvm::SmallString<8> outstr;
  raw_svector_ostream ostream(outstr);
  ostream.SetUnbuffered();

  legacy::FunctionPassManager function_pass_manager(module.get());
  legacy::PassManager module_pass_manager;

  // Gale
  module_pass_manager.add(createTargetTransformInfoWrapperPass(
      target_machine->getTargetIRAnalysis()));
  function_pass_manager.add(createTargetTransformInfoWrapperPass(
       target_machine->getTargetIRAnalysis()));

  tick;

  const auto kFTZDenorms = 1;

  // Insert a module flag for the FTZ handling.
  // module->addModuleFlag(llvm::Module::Override, "nvvm-reflect-ftz",
  //                      kFTZDenorms);

  //if (kFTZDenorms) {
  //  for (llvm::Function &fn : *module) {
  //    fn.addFnAttr("nvptx-f32ftz", "true");
  //  }
  //}

  PassManagerBuilder b;
  b.OptLevel = 3;
  b.Inliner = createFunctionInliningPass(b.OptLevel, 0, false);
  b.LoopVectorize = false;
  b.SLPVectorize = false;

  target_machine->adjustPassManager(b);

  b.populateFunctionPassManager(function_pass_manager);
  b.populateModulePassManager(module_pass_manager);

  // Override default to generate verbose assembly.
  target_machine->Options.MCOptions.AsmVerbose = true;

  // Output string stream

  // Ask the target to add backend passes as necessary.
  //bool fail = target_machine->addPassesToEmitFile(
  //    module_pass_manager, ostream, nullptr, llvm::CGFT_AssemblyFile, true);
  bool fail = target_machine->addPassesToEmitFile(
        module_pass_manager, ostream, nullptr, llvm::CGFT_ObjectFile, true);

  tickv(fail);
  tick;

  TI_ERROR_IF(fail, "Failed to set up passes to emit PTX source\n");

  {
    TI_PROFILER("llvm_function_pass");
    function_pass_manager.doInitialization();
    tick;
    for (llvm::Module::iterator i = module->begin(); i != module->end(); i++)
      function_pass_manager.run(*i);

    tick;
    function_pass_manager.doFinalization();
  }
  tick;

  {
    tick;
    module_pass_manager.run(*module);
  }
  tick;

  if (this->config_->print_kernel_llvm_ir_optimized) {
    static FileSequenceWriter writer(
        "taichi_kernel_amdgpu_llvm_ir_optimized_{:04d}.ll",
        "optimized LLVM IR (AMDGPU)");
    writer.write(module.get());
  }

  std::string buffer(outstr.begin(), outstr.end());

  // Null-terminate the ptx source
  buffer.push_back(0);
  tick;
  return buffer;
}

std::unique_ptr<JITSession> create_llvm_jit_session_cuda(
    TaichiLLVMContext *tlctx,
    CompileConfig *config,
    Arch arch) {
  TI_ASSERT(arch == Arch::cuda);
  // https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#data-layout
  auto data_layout = llvm::DataLayout(
    "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-ni:7");
  return std::make_unique<JITSessionCUDA>(tlctx, config, data_layout);
}
#else
std::unique_ptr<JITSession> create_llvm_jit_session_cuda(
    TaichiLLVMContext *tlctx,
    CompileConfig *config,
    Arch arch) {
  TI_NOT_IMPLEMENTED
}
#endif

TLANG_NAMESPACE_END
