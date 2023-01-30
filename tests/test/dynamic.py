import taichi as ti
ti.init(arch=ti.amdgpu)

def test_dynamic():
    x = ti.field(ti.i32)
    y = ti.field(ti.i32, shape=())

    n = 128

    ti.root.dynamic(ti.i, n).place(x)

    @ti.kernel
    def count():
        for i in x:
            y[None] += 1

    x[n // 3] = 1

    count()

    assert y[None] == n // 3 + 1

test_dynamic()
