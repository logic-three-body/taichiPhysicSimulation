import taichi as ti
ti.init(arch=ti.gpu)

N = 16
dt = 1e-4
dx = 1 / N
rho = 4e1
NF = 2 * N ** 2   # number of faces
NV = (N + 1) ** 2 # number of vertices
E, nu = 4e4, 0.2  # Young's modulus and Poisson's ratio
mu, lam = E / 2 / (1 + nu), E * nu / (1 + nu) / (1 - 2 * nu) # Lame parameters
ball_pos, ball_radius = ti.Vector([0.5, 0.0]), 0.32
gravity = ti.Vector([0, -40])
damping = 12.5

pos = ti.Vector.field(2, float, NV)
vel = ti.Vector.field(2, float, NV)
f2v = ti.Vector.field(3, int, NF)  # ids of three vertices of each face
B = ti.Matrix.field(2, 2, float, NF)
W = ti.field(float, NF)
phi = ti.field(float, NF)  # potential energy of each face (Neo-Hookean)
U = ti.field(float, (), needs_grad=True)  # total potential energy
f = ti.Vector.field(2, float, NV)

@ti.kernel
def update_force():
    for i in range(NF):
        ia, ib, ic = f2v[i]
        a, b, c = pos[ia], pos[ib], pos[ic]

        D_i = ti.Matrix.cols([a - c, b - c])
        F = D_i @ B[i]
        F_it =  F.inverse().transpose()

        PF = mu*(F - F_it) + lam* ti.log(F.determinant())*F_it
        H = -W[i]* PF@B[i].transpose()

        fa = ti.Vector([H[0, 0], H[1, 0]])
        fb = ti.Vector([H[0, 1], H[1, 1]])
        fc = -fa-fb
        f[ia] += fa
        f[ib] += fb
        f[ic] += fc

@ti.kernel
def advance():
    for i in range(NV):
        acc = f[i] / (rho * dx ** 2)
        vel[i] += dt * (acc + gravity)
        vel[i] *= ti.exp(-dt * damping)
    for i in range(NV):
        # ball boundary condition:
        disp = pos[i] - ball_pos
        disp2 = disp.norm_sqr()
        if disp2 <= ball_radius ** 2:
            NoV = vel[i].dot(disp)
            if NoV < 0: vel[i] -= NoV * disp / disp2
        # rect boundary condition:
        cond = pos[i] < 0 and vel[i] < 0 or pos[i] > 1 and vel[i] > 0
        for j in ti.static(range(pos.n)):
           if cond[j]: vel[i][j] = 0
        pos[i] += dt * vel[i]

@ti.kernel
def init_pos():
    for i, j in ti.ndrange(N + 1, N + 1):
        k = i * (N + 1) + j
        pos[k] = ti.Vector([i, j]) / N * 0.25 + ti.Vector([0.45, 0.45])
        vel[k] = ti.Vector([0, 0])
    for i in range(NF):
        ia, ib, ic = f2v[i]
        a, b, c = pos[ia], pos[ib], pos[ic]
        B_i = ti.Matrix.cols([a - c, b - c])
        B[i] = B_i.inverse()
        W[i] = abs( B_i.determinant())

@ti.kernel
def init_mesh():
    for i, j in ti.ndrange(N, N):
        k = (i * N + j) * 2
        a = i * (N + 1) + j
        b = a + 1
        c = a + N + 2
        d = a + N + 1
        f2v[k + 0] = [a, b, c]
        f2v[k + 1] = [c, d, a]

@ti.kernel
def init_parameter():
    for i in f:
        f[i] = ti.Vector([0.0, 0.0])

init_mesh()
init_pos()
gui = ti.GUI('FEM99')
while gui.running:
    for e in gui.get_events():
        if e.key == gui.ESCAPE:
            gui.running = False
        elif e.key == 'r':
            init_pos()
    for i in range(30):
        init_parameter()
        update_force()
        advance()
    gui.circles(pos.to_numpy(), radius=2, color=0xffaa33)
    gui.circle(ball_pos, radius=ball_radius * 512, color=0x666666)
    gui.show()

    # remark
    """
   Lec3的课件lec3-elasticity中Neo-Hookean的P(F)错了，应该是 F - F-T
   附一下基于demo FEM99的手动求导版本 ，感谢胡老师超棒的课程和超棒的taichi : )
    """