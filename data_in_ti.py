#Play Fields
import taichi as ti
ti.init(arch=ti.gpu)
a=ti.field(dtype=ti.f32,shape=(42,63))
#A 42x63 scalar field
b=ti.Vector.field(3,dtype=ti.f32,shape=4)
#B 4-elems field of 3D vectors
c=ti.Matrix.field(2,2,dtype=ti.f32,shape=(3,5))
#C 3x5 field of 2x2 matrices
loss=ti.field(dtype=ti.f32,shape=())
# (0 D) field of a single scalar

a[3,4]=1
print('a[3,4]=' ,a[3,4])
print('a[4,3]='  ,a[4,3])
print('a[None]=',a[None])
print('a=',a)

a=4
print('a=',a) #a=4
#print('a[None]=',a[None]) #a is not a field anymore

b[2]=[5,7,8]
print('b=',b)
#print('b[0]='. b[0][0],b[0][1],b[0][2]) #wrong
print('b[0]=',b[0]) #<taichi.lang.matrix.Matrix.Proxy object at 0x000001D365CCE4C0>
print('b[0][1]=',b[0][1]) #0.0
print('b[2]=',b[2]) #<taichi.lang.matrix.Matrix.Proxy object at 0x000001D365CCE4C0>

#c[3,5]=[2,2]
c[1,1]=ti.Matrix([[2, 3], [4, 5]])
print('c=',c)
print('c[1,1][0]=',c[1,1][0])#2.0
print('c[1,1][0]=',c[1,1][1])#4.0
print('c[1,1][0]=',c[1,1][1,1])#5.0
print('c[1,1,1,1]=',c[1,1,1,1])#<taichi.lang.matrix.Matrix.Proxy object at 0x000001591CFA7A90>
print('c[1,1,1,1,0]=',c[1,1,1,1,0])#<taichi.lang.matrix.Matrix.Proxy object at 0x000001591CFA7A90>

c[0,1]=ti.Matrix([[9, 9], [8, 8],[7,8]])
print('c[0,1][2]=',c[0,1][0,1])#3
#print('c[0,1][2]=',c[0,1][0,2])#wrong
print('c',c)

print('loss=',loss)
print('loss[None]=',loss[None])
loss[None]=3
print('loss=',loss)
print('loss[None]=',loss[None])

"""
wrong
"""
#loss[0]=3
#print('loss[0]=',loss[0])