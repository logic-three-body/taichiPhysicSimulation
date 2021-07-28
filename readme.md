# 单摆模型

实验环境：[taichi](https://www.bilibili.com/video/av330106419?p=1)

参考：[【官方双语】微分方程概论-第一章_哔哩哔哩_bilibili](https://www.bilibili.com/video/av50290975)

## 思路

主要是求解常微分方程以获取角度，涉及到**相空间**的概念

3b1视频末尾的求解代码：

```python
import numpy as np

# Physical constants
g = 9.8
L = 2 #单摆长
mu = 0.1

THETA_0 = np.pi / 3  # 60 degrees
THETA_DOT_0 = 0  # No initial angular velocity

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
```

## 基于taichi的实现

```python
import taichi as ti
import numpy as np
max_num_dots=2
# Physical constants
g = 9.8
L = 0.5 #单摆长 要更改
mu = 0.1

THETA_0 = np.pi / 6 # 30 degrees 分母 0.25以下单摆会不动 1也不会动 1.001以下也不动 （1.01可以）
THETA_DOT_0 = 0  # No initial angular velocity
ti.init(arch=ti.cpu)
x=ti.Vector.field(2,dtype=ti.f32,shape=max_num_dots)#位置


# Definition of ODE
def get_theta_double_dot(theta, theta_dot):
    return -mu * theta_dot - (g / L) * np.sin(theta)

def main():
    gui=ti.GUI('pendulum test',background_color=0xfaf3e0)



    delta_t = 0.01  # Some time step
    t=10000#时间范围
    theta = THETA_0#初始化变化角
    theta_dot = THETA_DOT_0
    x[0]=ti.Vector([0.5,0.9])#悬挂点
    for time in np.arange(0, t, delta_t):#core
            x[1]=ti.Vector([x[0][0]-L*ti.sin(theta),x[0][1]-L*ti.cos(theta)])#摆动点
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
```

浮点数误差对结果影响非常大（未设阻力，理想情况下应该每一个周期相对中轴线摆幅一致，但最后会越摆越高，能量不守恒）

![单摆2](https://i.loli.net/2021/07/28/gmwMoeubHifLEyK.gif)

![单摆1](https://i.loli.net/2021/07/28/wWcSe57QtA9qTp3.gif)