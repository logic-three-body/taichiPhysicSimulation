import taichi as ti
ti.init(arch=ti.cuda)
field=ti.field(dtype=ti.f32,shape=(4,8,16,32,64))

@ti.kernel
def print_shape(x:ti.template()):
    ti.static_print(x.shape)
    for i in ti.static(range(len(x.shape))):
        print(x.shape[i])

print_shape(field)
