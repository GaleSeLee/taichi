import taichi as ti
ti.init(arch=ti.amdgpu, print_kernel_llvm_ir=True)

f = ti.Matrix.field(3, 3, ti.f32, shape=())

@ti.kernel
def foo():
    f[None][0, 0] = 1.0

foo()
