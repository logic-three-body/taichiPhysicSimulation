import taichi as ti
ti.init(arch=ti.cpu)
m=1#默认质量为1
max_num_dots=2
x=ti.Vector.field(2,dtype=ti.f32,shape=max_num_dots)#位置
v=ti.Vector.field(2,dtype=ti.f32,shape=max_num_dots)#速度
f=ti.Vector.field(2,dtype=ti.f32,shape=max_num_dots)#拉力
fixed=ti.field(dtype=ti.i32,shape=max_num_dots)#bool 是否为固定点
#暂时为弹簧
springY=5
rest_len=0.1


PI=180.0

num_dots=ti.field(dtype=ti.i32,shape=())
dt=1e-3
substeps = 10#步长帧
@ti.kernel
def substep():
    n=num_dots[None]
    grav=ti.Vector([0, -9.8])
    for i in range(n):
        if not fixed[i]:#是否为固定点
            f[i]=grav
            for j in range(n):
                    #spring
                    x_ij = x[i] - x[j]



                    print('thea=',thea)
                    print('sinthea',ti.sin(thea))
                    f[i]
                    print('n=',n)
                    print('x[i]',x[i])
                    print('x[j]',x[j])

                    print('f[',i,']=',f[i])
                    print('x[',i,j,']=',x_ij)



                    v[i]+=dt*f[i]
                    x[i]+=dt*v[i]


            # Collide with four walls
            for d in ti.static(range(2)):
                # d = 0: treating X (horizontal) component
                # d = 1: treating Y (vertical) component

                if x[i][d] < 0:  # Bottom and left
                    x[i][d] = 0  # move particle inside
                    v[i][d] = 0  # stop it from moving further

                if x[i][d] > 1:  # Top and right
                    x[i][d] = 1  # move particle inside
                    v[i][d] = 0  # stop it from moving further

@ti.kernel
def add_dot(pos_x:ti.f32,pos_y:ti.f32,fixed_:ti.i32):
    new_dot_id=num_dots[None]
    num_dots[None]+=1
    #print('posx,posy',pos_x,pos_y)
    x[new_dot_id]=ti.Vector([pos_x,pos_y])
    #print('x id',new_dot_id,x[new_dot_id])
    fixed[new_dot_id]=fixed_

def main():
    gui=ti.GUI('Pendulum System',background_color=0xfaf3e0)
    while True:
        for i in range(substeps):
            substep()
        if num_dots[None]<max_num_dots:#局限为两个点
            for e in gui.get_events(ti.GUI.PRESS):
                if e.key==ti.GUI.LMB:
                    add_dot(e.pos[0],e.pos[1],bool(gui.is_pressed(ti.GUI.SHIFT)))
                    #print(e.pos[0],e.pos[1])

        X=x.to_numpy()

        for i in range(num_dots[None]):
            for j in range(num_dots[None]):
                gui.line(begin=X[i],end=X[j],color=0xff75a0,radius=2)


        for i in range(num_dots[None]):
            if fixed[i]:#固定点
                c=0xe40017
                r=3
            else:#非固定点
                c=0xb68973
                r=10
            gui.circle(X[i],color=c,radius=r)

        gui.show()


if __name__ == '__main__':
        main()