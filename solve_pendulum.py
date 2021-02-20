import taichi as ti
ti.init(arch=ti.cpu)

max_num_dots=1024
x=ti.Vector.field(2,dtype=ti.f32,shape=max_num_dots)
v=ti.Vector.field(2,dtype=ti.f32,shape=max_num_dots)
fixed=ti.field(dtype=ti.i32,shape=max_num_dots)#bool

num_dots=ti.field(dtype=ti.i32,shape=())

@ti.kernel
def add_dot(pos_x:ti.f32,pos_y:ti.f32,fixed_:ti.f32):
    new_dot_id=num_dots[None]
    num_dots[None]+=1

    x[new_dot_id]=ti.Vector([pos_x,pos_y])
    fixed[new_dot_id]=fixed_

def main():
    gui=ti.GUI('Pendulum System',background_color=0xfaf3e0)
    while True:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key==ti.GUI.LMB:
                add_dot(e.pos[0],e.pos[1],False)

        X=x.to_numpy()
        for i in range(num_dots[None]):
            gui.circle(X[i],color=0x1e212d,radius=5)

        gui.show()


if __name__ == '__main__':
        main()