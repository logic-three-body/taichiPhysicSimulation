import taichi as ti
ti.init(arch=ti.cpu)

max_num_dots=1024
x=ti.Vector.field(2,dtype=ti.f32,shape=max_num_dots)
v=ti.Vector.field(2,dtype=ti.f32,shape=max_num_dots)
fixed=ti.field(dtype=ti.i32,shape=max_num_dots)#bool 是否为固定点

num_dots=ti.field(dtype=ti.i32,shape=())
dt=1e-3
substeps = 10#步长帧
@ti.kernel
def substep():
    n=num_dots[None]

    for i in range(n):
        if not fixed[i]:#是否为固定点
            v[i]+=dt*ti.Vector([0,-9.8])
            x[i]+=dt*v[i]

@ti.kernel
def add_dot(pos_x:ti.f32,pos_y:ti.f32,fixed_:ti.i32):
    new_dot_id=num_dots[None]
    num_dots[None]+=1

    x[new_dot_id]=ti.Vector([pos_x,pos_y])
    fixed[new_dot_id]=fixed_

def main():
    gui=ti.GUI('Pendulum System',background_color=0xfaf3e0)
    while True:
        for i in range(substeps):
            substep()

        for e in gui.get_events(ti.GUI.PRESS):
            if e.key==ti.GUI.LMB:
                add_dot(e.pos[0],e.pos[1],bool(gui.is_pressed(ti.GUI.SHIFT)))

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