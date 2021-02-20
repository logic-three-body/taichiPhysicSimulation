import taichi as ti
ti.init(arch=ti.gpu)
""""
@ti.kernel
def hello(i:ti.i32):
    a=40
    print('Hello',a+i)

hello(2)

@ti.kernel
def calc()->ti.int32:
    s=0
    for i in range(10):
        s+=i
    return s

print (calc())

@ti.func
def triple(x):
    return x*3

@ti.kernel# ?⬇
def triple_A():
    for i in range(128):
        a[i]=triple(a[i])


a=1
b=2
print(a/b)
print(a//b)

#print(ti.random(dtype=ti.i32))
#print(ti.random(dtype=ti.f32))
"""

"""""
# Taichi 作用域
v0 = ti.Vector([1.0, 2.0, 3.0])
v1 = ti.Vector([4.0, 5.0, 6.0])
v2 = ti.Vector([7.0, 8.0, 9.0])

# 指定行中的数据
a = ti.Matrix.rows([v0, v1, v2])
print(a)
# 指定列中的数据
a = ti.Matrix.cols([v0, v1, v2])
print(a)
# 可以用列表代替参数中的向量
a = ti.Matrix.rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
print(a)
"""
n=320
pixels=ti.field(dtype=ti.f32,shape=(n*2,n))
@ti.kernel
def paint(t:ti.f32):
    for i,j in pixels :
        pixels[i,j]=i*0.001+j*0.002+t

gui = ti.GUI("Julia Set", res=(n * 2, n))
for i in range(100):
    paint(i * 0.03)
    gui.set_image(pixels)
    gui.show()