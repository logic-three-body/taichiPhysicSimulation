import taichi as ti
import numpy as np
max_num_dots=2
# Physical constants
g = 9.8
L = 0.5 #单摆长 要更改
mu = 0.1

THETA_0 = np.pi / 0.3  # 30 degrees
THETA_DOT_0 = 0  # No initial angular velocity
ti.init(arch=ti.cpu)
x=ti.Vector.field(2,dtype=ti.f32,shape=max_num_dots)#位置


# Definition of ODE
def get_theta_double_dot(theta, theta_dot):
    return -mu * theta_dot - (g / L) * np.sin(theta)


# Solution to the differential equation
def theta(t):
    # Initialize changing values
    theta = THETA_0
    theta_dot = THETA_DOT_0
    delta_t = 0.01  # Some time step
    for time in np.arange(0, t, delta_t):
        # Take many little time steps of size delta_t
        # until the total time is the function's input
        theta_double_dot = get_theta_double_dot(
            theta, theta_dot
        )
        theta += theta_dot * delta_t
        theta_dot += theta_double_dot * delta_t
    return theta







def main():
    gui=ti.GUI('pendulum test',background_color=0xfaf3e0)



    delta_t = 0.01  # Some time step
    t=1000#时间范围
    theta = THETA_0#初始化变化角
    theta_dot = THETA_DOT_0
    x[0]=ti.Vector([0.5,0.9])#悬挂点
    for time in np.arange(0, t, delta_t):#core
            x[1]=ti.Vector([x[0][0]-L*ti.sin(theta),x[0][1]-L*ti.cos(theta)])#摆动点
            #x[1]=ti.Vector([x[0][0]+0.1,x[0][1]-0.6])
            print(x)
            print(x[1][0],x[1][1])
            gui.circle(x[0],color=0xe40017,radius=2)
            gui.circle(x[1],color=0xb68973,radius=10)
            gui.line(begin=x[0], end=x[1], color=0xff75a0, radius=2)
            gui.show()
            theta_double_dot = get_theta_double_dot(#update
                theta, theta_dot
            )
            theta += theta_dot * delta_t
            theta_dot += theta_double_dot * delta_t

if __name__ == '__main__':
    main()