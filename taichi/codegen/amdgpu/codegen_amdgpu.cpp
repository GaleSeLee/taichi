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
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/rhi/cuda/cuda_context.h"
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
        auto input = llvm_val[stmt->operand];
        auto input_taichi_type = stmt->operand->ret_type;
        auto op = stmt->op_type;

// TODO (Gale)
// add int type support
#define UNARY_STD(x)                                                          \
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
        UNARY_STD(cos);
        UNARY_STD(acos);
        UNARY_STD(sin);
        UNARY_STD(asin);
        UNARY_STD(tan);
        UNARY_STD(tanh);
        UNARY_STD(exp);
        UNARY_STD(log);
        UNARY_STD(sqrt);
        else {
            TI_P(unary_op_type_name(op));
            TI_NOT_IMPLEMENTED
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

    
}