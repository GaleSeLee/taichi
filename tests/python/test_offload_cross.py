import taichi as ti
from tests import test_utils


@test_utils.test()
def test_offload_with_cross_block_locals():
    ret = ti.field(ti.f32)

    ti.root.place(ret)

    @ti.kernel
    def ker():
        s = 0
        for i in range(10):
            s += i
        ret[None] = s

    ker()

    assert ret[None] == 45


@test_utils.test()
def test_offload_with_cross_block_locals2():
    ret = ti.field(ti.f32)

    ti.root.place(ret)

    @ti.kernel
    def ker():
        s = 0
        for i in range(10):
            s += i
        ret[None] = s
        s = ret[None] * 2
        for i in range(10):
            ti.atomic_add(ret[None], s)

    ker()

    assert ret[None] == 45 * 21


@test_utils.test()
def test_offload_with_cross_block_locals3():
    ret = ti.field(ti.f32, shape=())

    @ti.kernel
    def ker():
        s = 1
        t = s
        for i in range(10):
            s += i
        ret[None] = t

    ker()

    assert ret[None] == 1


@test_utils.test()
def test_offload_with_cross_block_locals4():
    ret = ti.field(ti.f32, shape=())

    @ti.kernel
    def ker():
        a = 1
        b = 0
        for i in range(10):
            b += a
        ret[None] = b

    ker()

    assert ret[None] == 10


@test_utils.test()
def test_offload_with_flexible_bounds():
    s = ti.field(ti.i32, shape=())
    lower = ti.field(ti.i32, shape=())
    upper = ti.field(ti.i32, shape=())

    @ti.kernel
    def ker():
        for i in range(lower[None], upper[None]):
            s[None] += i

    lower[None] = 10
    upper[None] = 20
    ker()

    assert s[None] == 29 * 10 // 2


@test_utils.test()
def test_offload_with_cross_block_globals():
    ret = ti.field(ti.f32)

    ti.root.place(ret)

    @ti.kernel
    def ker():
        ret[None] = 0
        for i in range(10):
            ret[None] += i
        ret[None] += 1

    ker()

    assert ret[None] == 46


@test_utils.test()
def test_offload_with_cross_nested_for():
    @ti.kernel
    def run(a: ti.i32):
        b = a + 1
        for x in range(1):
            for i in range(b):
                pass

    run(2)


@test_utils.test()
def test_offload_with_cross_if_inside_for():
    @ti.kernel
    def run(a: ti.i32):
        b = a > 2
        for x in range(1):
            if b:
                pass

    run(2)
