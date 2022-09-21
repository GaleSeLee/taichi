#include "taichi/codegen/amdgpu/codegen_amdgpu.h"

#include <vector>
#include <set>
#include <functional>

#include "taichi/common/core.h"
#include "taichi/util/io.h"
#include "taichi/util/statistics.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/program/program.h"
#include "taichi/util/lang_util.h"
#include "taichi/rhi/amdgpu/amdgpu_driver.h"
#include "taichi/rhi/amdgpu/amdgpu_context.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#include "taichi/util/action_recorder.h"
#include "taichi/analysis/offline_cache_util.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/transforms.h"

TLANG_NAMESPACE_BEGIN

using namespace llvm;

class TaskCodeGenAMDGPU : public TaskCodeGenLLVM {
public:
    using IRVisitor::visit;
    TaskCodeGenAMDGPU(Kernel *kernel, IRNode *ir = nullptr)
        : TaskCodeGenLLVM(kernel, ir) {}

    llvm::Value *create_print(std::string tag,
                                DataType dt,
                                llvm::Value *value) override {
        TI_NOT_IMPLEMENTED
    }

    llvm::Value *create_print(const std::string &format,
                        const std::vector<llvm::Type *> &types,
                        const std::vector<llvm::Value *> &values) {
        TI_NOT_IMPLEMENTED
    }

    void visit(PrintStmt *stmt) override {
        TI_NOT_IMPLEMENTED
    }

    void emit_extra_unary(UnaryOpStmt *stmt) override {
        auto input_taichi_type = stmt->operand->ret_type;
        auto op = stmt->op_type;

// TODO (Gale)
// add int type support
#define UNARY_STD(x)                                                      \
else if (op == UnaryOpType::x) {                                          \
    if (input_taichi_type->is_primitive(PrimitiveTypeID::f16)) {          \
        llvm_val[stmt] = create_call("__ocml_" #x "_f16");                \
    } else if (input_taichi_type->is_primitive(PrimitiveTypeID::f32)) {   \
        llvm_val[stmt] = create_call("__ocml_" #x "_f32");                \
    } else if (input_taichi_type->is_primitive(PrimitiveTypeID::f64)) {   \
        llvm_val[stmt] = create_call("__ocml_" #x "_f64");                \
    } else {                                                              \
        TI_NOT_IMPLEMENTED                                                \   
    }                                                                     \
}
        if (op == UnaryOpType::abs) {
            if (input_taichi_type->is_primitive(PrimitiveTypeID::f16)) {          
                llvm_val[stmt] = create_call("__ocml_fasb_f16");                
            } else if (input_taichi_type->is_primitive(PrimitiveTypeID::f32)) {   
                llvm_val[stmt] = create_call("__ocml_fabs_f32");                
            } else if (input_taichi_type->is_primitive(PrimitiveTypeID::f64)) {   
                llvm_val[stmt] = create_call("__ocml_fabs_f64");                
            } else {                                                              
                TI_NOT_IMPLEMENTED
            }
        }
        UNARY_STD(cos)
        UNARY_STD(acos)
        UNARY_STD(sin)
        UNARY_STD(asin)
        UNARY_STD(tan)
        UNARY_STD(tanh)
        UNARY_STD(exp)
        UNARY_STD(log)
        UNARY_STD(sqrt)
        else {
            TI_NOT_IMPLEMENTED
            TI_P(unary_op_type_name(op));
        }
#undef UNARY_STD
    }

    llvm::Value *optimized_reduction(AtomicOpStmt *stmt) override {
        // TODO (Gale)
        // Check
        if (!stmt->is_reduction) {
            return nullptr;
        }
        TI_ASSERT(stmt->val->ret_type->is<PrimitiveType>());
        PrimitiveTypeID prim_type =
            stmt->val->ret_type->cast<PrimitiveType>()->type;

        std::unordered_map<PrimitiveTypeID,
                            std::unordered_map<AtomicOpType, std::string>>
            fast_reductions;

        fast_reductions[PrimitiveTypeID::i32][AtomicOpType::add] = "reduce_add_i32";
        fast_reductions[PrimitiveTypeID::f32][AtomicOpType::add] = "reduce_add_f32";
        fast_reductions[PrimitiveTypeID::i32][AtomicOpType::min] = "reduce_min_i32";
        fast_reductions[PrimitiveTypeID::f32][AtomicOpType::min] = "reduce_min_f32";
        fast_reductions[PrimitiveTypeID::i32][AtomicOpType::max] = "reduce_max_i32";
        fast_reductions[PrimitiveTypeID::f32][AtomicOpType::max] = "reduce_max_f32";

        fast_reductions[PrimitiveTypeID::i32][AtomicOpType::bit_and] =
            "reduce_and_i32";
        fast_reductions[PrimitiveTypeID::i32][AtomicOpType::bit_or] =
            "reduce_or_i32";
        fast_reductions[PrimitiveTypeID::i32][AtomicOpType::bit_xor] =
            "reduce_xor_i32";

        AtomicOpType op = stmt->op_type;
        if (fast_reductions.find(prim_type) == fast_reductions.end()) {
            return nullptr;
        }
        TI_ASSERT(fast_reductions.at(prim_type).find(op) !=
                    fast_reductions.at(prim_type).end());
        return create_call(fast_reductions.at(prim_type).at(op),
                            {llvm_val[stmt->dest], llvm_val[stmt->val]});
    }

#ifndef TI_LLVM_15
    llvm::Value *atomic_op_using_cas(
        llvm::Value *output_address,
        llvm::Value *val,
        std::function<llvm::Value *(llvm::Value *, llvm::Value *)> op) override {
            TI_NOT_IMPLEMENTED
        }
#endif // TI_LLVM_15

    void visit(RangeForStmt *for_stmt) override {
        create_naive_range_for(for_stmt);
    }

    void create_offload_range_for(OffloadedStmt *stmt) override {
        auto tls_prologue = create_xlogue(stmt->tls_prologue);

        llvm::Function *body;
        {
            auto guard = get_function_creation_guard(
                {llvm::PointerType::get(get_runtime_type("RuntimeContext"), 0),
                get_tls_buffer_type(), tlctx->get_data_type<int>()});

            auto loop_var = create_entry_block_alloca(PrimitiveType::i32);
            loop_vars_llvm[stmt].push_back(loop_var);
            builder->CreateStore(get_arg(2), loop_var);
            stmt->body->accept(this);

            body = guard.body;
        }

        auto epilogue = create_xlogue(stmt->tls_epilogue);

        auto [begin, end] = get_range_for_bounds(stmt);
        create_call("gpu_parallel_range_for",
                    {get_arg(0), begin, end, tls_prologue, body, epilogue,
                    tlctx->get_constant(stmt->tls_size)});
    }
 
    void create_offload_mesh_for(OffloadedStmt *stmt) override {
        TI_NOT_IMPLEMENTED
    }

    void emit_amdgpu_gc(OffloadedStmt *stmt) {
        auto snode_id = tlctx->get_constant(stmt->snode->id);
        {
            init_offloaded_task_function(stmt, "gather_list");
            call("gc_parallel_0", get_context(), snode_id);
            finalize_offloaded_task_function();
            current_task->grid_dim = prog->config.saturating_grid_dim;
            current_task->block_dim = 64;
            offloaded_tasks.push_back(*current_task);
            current_task = nullptr;
        }
        {
            init_offloaded_task_function(stmt, "reinit_lists");
            call("gc_parallel_1", get_context(), snode_id);
            finalize_offloaded_task_function();
            current_task->grid_dim = 1;
            current_task->block_dim = 1;
            offloaded_tasks.push_back(*current_task);
            current_task = nullptr;
        }
        {
            init_offloaded_task_function(stmt, "zero_fill");
            call("gc_parallel_2", get_context(), snode_id);
            finalize_offloaded_task_function();
            current_task->grid_dim = prog->config.saturating_grid_dim;
            current_task->block_dim = 64;
            offloaded_tasks.push_back(*current_task);
            current_task = nullptr;
        }
    }

    bool kernel_argument_by_val() const override {
        return false;  // on AMDGPU, pass the argument by value is not allowed
    }

    void visit(GlobalLoadStmt *stmt) override {
        if (auto get_ch = stmt->src->cast<GetChStmt>()) {
            bool should_cache_as_read_only = current_offload->mem_access_opt.has_flag(
                get_ch->output_snode, SNodeAccessFlag::read_only);
            create_global_load(stmt, should_cache_as_read_only);
        } else {
            create_global_load(stmt, false);
        }
    }

    void create_bls_buffer(OffloadedStmt *stmt) {
        TI_NOT_IMPLEMENTED
    }

    void visit(OffloadedStmt *stmt) override {
        stat.add("codegen_offloaded_tasks");
        if (stmt->bls_size > 0)
            create_bls_buffer(stmt);
        TI_ASSERT(current_offload == nullptr);
        current_offload = stmt;
        using Type = OffloadedStmt::TaskType;
        if (stmt->task_type == Type::gc) {
            std::cout << "[GALE] GC" << std::endl;
            emit_amdgpu_gc(stmt);
        } else {
            init_offloaded_task_function(stmt);
            if (stmt->task_type == Type::serial) {
                stmt->body->accept(this);
            } else if (stmt->task_type == Type::range_for) {
                create_offload_range_for(stmt);
            } else if (stmt->task_type == Type::struct_for) {
                create_offload_struct_for(stmt, true);
            } else if (stmt->task_type == Type::mesh_for) {
                create_offload_mesh_for(stmt);
            } else if (stmt->task_type == Type::listgen) {
                emit_list_gen(stmt);
            } else {
                TI_NOT_IMPLEMENTED
            }
            finalize_offloaded_task_function();
            current_task->grid_dim = stmt->grid_dim;
            if (stmt->task_type == Type::range_for) {
                if (stmt->const_begin && stmt->const_end) {
                    int num_threads = stmt->end_value - stmt->begin_value;
                    int grid_dim = ((num_threads % stmt->block_dim) == 0)
                                        ? (num_threads / stmt->block_dim)
                                        : (num_threads / stmt->block_dim) + 1;
                    grid_dim = std::max(grid_dim, 1);
                    current_task->grid_dim = std::min(stmt->grid_dim, grid_dim);
                }
            }
            if (stmt->task_type == Type::listgen) {
                // Note: 32 is a temporary number
                int query_max_block_per_sm = 32;
                // AMDGPUDriver::get_instance().device_get_attribute(
                //     &query_max_block_per_sm,
                //     HIP_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, nullptr);
                int num_SMs;
                AMDGPUDriver::get_instance().device_get_attribute(
                    &num_SMs, HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0);
                current_task->grid_dim = num_SMs * query_max_block_per_sm;
            }
            current_task->block_dim = stmt->block_dim;
            TI_ASSERT(current_task->grid_dim != 0);
            TI_ASSERT(current_task->block_dim != 0);
            offloaded_tasks.push_back(*current_task);
            current_task = nullptr;
        }
        current_offload = nullptr;
    }

    void visit(ExternalFuncCallStmt *stmt) override {
        if (stmt->type == ExternalFuncCallStmt::BITCODE) {
            TaskCodeGenLLVM::visit_call_bitcode(stmt);
        } else {
            TI_NOT_IMPLEMENTED
        }
    }

    void visit(ExternalTensorShapeAlongAxisStmt *stmt) override {
        const auto arg_id = stmt->arg_id;
        const auto axis = stmt->axis;
        llvm_val[stmt] = create_call("RuntimeContext_get_extra_args",
                                     {get_context(), tlctx->get_constant(arg_id),
                                     tlctx->get_constant(axis)});
    }
    
    void visit(BinaryOpStmt *stmt) override {
        auto op = stmt->op_type;
        auto ret_taichi_type = stmt->ret_type;
        if (op != BinaryOpType::atan2 && op != BinaryOpType::pow) {
            return TaskCodeGenLLVM::visit(stmt);
        }
        auto lhs = llvm_val[stmt->lhs];
        auto rhs = llvm_val[stmt->rhs];

#define BINARY_STD(x)                                                           \
    if (op == BinaryOpType::x) {                                                \
        if (ret_taichi_type->is_primitive(PrimitiveTypeID::f16)) {            \
            llvm_val[stmt] = create_call("__ocml_" #x "_16f", {lhs, rhs});        \
        } else if (ret_taichi_type->is_primitive(PrimitiveTypeID::f32)) {     \
            llvm_val[stmt] = create_call("__ocml_" #x "_32f", {lhs, rhs});        \
        } else if (ret_taichi_type->is_primitive(PrimitiveTypeID::i64)) {     \
            llvm_val[stmt] = create_call("__ocml_" #x "_64f", {lhs, rhs});        \
        } else {                                                                \
            TI_NOT_IMPLEMENTED                                                  \
        }                                                                       \
    }
        BINARY_STD(pow);
        BINARY_STD(atan2);
#undef BINARY_STD
    }
};

#ifdef TI_WITH_LLVM
// static
std::unique_ptr<TaskCodeGenLLVM> KernelCodeGenAMDGPU::make_codegen_llvm(
    Kernel *kernel,
    IRNode *ir) {
    return std::make_unique<TaskCodeGenAMDGPU>(kernel, ir);
}
#endif  // TI_WITH_LLVM

LLVMCompiledData KernelCodeGenAMDGPU::compile_task(
        std::unique_ptr<llvm::Module> &&module,
        OffloadedStmt *stmt) {
    TaskCodeGenAMDGPU gen(kernel, stmt);
    return gen.run_compilation();
}

FunctionType KernelCodeGenAMDGPU::compile_to_function() {
    auto *llvm_prog = get_llvm_program(prog);
    auto *tlctx = llvm_prog->get_llvm_context(kernel->arch);

    std::vector<LLVMCompiledData> data = compile_kernel_to_module();
    AMDGPUModuleToFunctionConverter converter{tlctx,
                                        llvm_prog->get_runtime_executor()};

    return converter.convert(this->kernel, std::move(data));
}

FunctionType AMDGPUModuleToFunctionConverter::convert(
        const std::string &kernel_name,
        const std::vector<LlvmLaunchArgInfo> &args,
        std::vector<LLVMCompiledData> &&data) const {
    std::vector<JITModule *> amdgpu_modules;
    std::vector<std::vector<OffloadedTask> > offloaded_tasks;
    amdgpu_modules.reserve(data.size());
    for (auto &datum : data) {
        auto &mod = datum.module;
        auto &tasks = datum.tasks;
        for (const auto &task : tasks) {
            llvm::Function *func = mod->getFunction(task.name);
            TI_ASSERT(func);
            func->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
        }
        auto jit = tlctx_->jit.get();
        amdgpu_modules.push_back(
            jit->add_module(std::move(mod), executor_->get_config()->gpu_max_reg));
        offloaded_tasks.push_back(std::move(tasks));
    }
    return [amdgpu_modules, kernel_name, args, offloaded_tasks,
                executor = this->executor_](RuntimeContext &context) {
            AMDGPUContext::get_instance().make_current();
            std::vector<void *> arg_buffers(args.size(), nullptr);
            std::vector<void *> device_buffers(args.size(), nullptr);
            bool transferred = false;
            for (int i = 0; i < (int)args.size(); i++) {
                if (args[i].is_array) {
                    const auto arr_sz = context.array_runtime_sizes[i];
                    if (arr_sz == 0) continue;
                    arg_buffers[i] = context.get_arg<void *>(i);
                    if (context.device_allocation_type[i] == 
                        RuntimeContext::DevAllocType::kNone) {
                            unsigned int attr_val = 0;
                            uint32_t ret_code = AMDGPUDriver::get_instance().mem_get_attribute.call(
                                &attr_val, HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                (void *)arg_buffers[i]);

                            if (ret_code != HIP_SUCCESS || attr_val != HIP_MEMORYTYPE_DEVICE) {
                                transferred = true;
                                AMDGPUDriver::get_instance().malloc(&device_buffers[i], arr_sz);
                                AMDGPUDriver::get_instance().memcpy_host_to_device(
                                    (void *)device_buffers[i], arg_buffers[i], arr_sz);
                            } else {
                                device_buffers[i] = arg_buffers[i];
                            }

                            context.set_arg(i, (uint64)device_buffers[i]);
                    } else if (arr_sz > 0) { // why use arr_sz constrain?
                        DeviceAllocation *ptr =
                            static_cast<DeviceAllocation *>(arg_buffers[i]);
                        device_buffers[i] = executor->get_ndarray_alloc_info_ptr(*ptr);
                        arg_buffers[i] = device_buffers[i];
                        context.set_arg(i, (uint64)device_buffers[i]);
                    }
                }
            }
            if (transferred) {
                AMDGPUDriver::get_instance().stream_synchronize(nullptr);
            }

            void *context_pointer;
            AMDGPUDriver::get_instance().malloc(
                (void **)&context_pointer, sizeof(RuntimeContext));
            AMDGPUDriver::get_instance().memcpy_host_to_device(
                context_pointer, &context, sizeof(RuntimeContext));
            for (int i = 0; i < offloaded_tasks.size(); i++) {
                for (auto &task : offloaded_tasks[i]) {
                    TI_TRACE("Launching kernel {}<<<{}, {}>>>", task.name, task.grid_dim,
                                task.block_dim);
                    amdgpu_modules[i]->launch(task.name, task.grid_dim, task.block_dim, 0,
                                                (void *)&context_pointer, (int)sizeof(RuntimeContext *));
                }
            }
            // TODO (Gale)
            if (transferred) {
                AMDGPUDriver::get_instance().stream_synchronize(nullptr);
                for (int i = 0; i < args.size(); i++) {
                    if (device_buffers[i] != arg_buffers[i]) {
                        AMDGPUDriver::get_instance().memcpy_device_to_host(
                            arg_buffers[i], (void *)device_buffers[i],
                            context.array_runtime_sizes[i]);
                        AMDGPUDriver::get_instance().mem_free((void *)device_buffers[i]);
                    }
                }
            }
            TI_TRACE("Launching kernel");
            //AMDGPUDriver::get_instance().mem_free((void *)context_pointer);
            
        };
}

TLANG_NAMESPACE_END