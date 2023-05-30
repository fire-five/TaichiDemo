# http://viterbi-web.usc.edu/~jbarbic/femdefo/sifakis-courseNotes-TheoryAndDiscretization.pdf
import taichi as ti
import math


@ti.data_oriented
class CrossBeam:
    def __init__(self, N_x, N_y, dx):
        self.N_x = N_x
        self.N_y = N_y
        self.init_x = 0.1
        self.init_y = 0.6
        self.N = N_x * N_y
        self.N_edges = (N_x-1)*N_y + N_x*(N_y - 1) + (N_x-1) * \
            (N_y-1)  # horizontal + vertical + diagonal springs
        self.N_triangles = 2 * (N_x-1) * (N_y-1)
        self.dx = dx
        # simulation components
        self.x = ti.Vector.field(2, ti.f32, self.N)
        self.v = ti.Vector.field(2, ti.f32, self.N)
        self.grad = ti.Vector.field(2, ti.f32, self.N)
        self.elements_Dm_inv = ti.Matrix.field(2, 2, ti.f32, self.N_triangles)
        self.elements_V0 = ti.field(ti.f32, self.N_triangles)
        # physical quantities
        self.m = 1
        self.g = 9.8
        self.YoungsModulus = ti.field(ti.f32, ())
        self.PoissonsRatio = ti.field(ti.f32, ())
        self.LameMu = ti.field(ti.f32, ())
        self.LameLa = ti.field(ti.f32, ())

        # geometric components
        self.triangles = ti.Vector.field(3, ti.i32, self.N_triangles)
        self.edges = ti.Vector.field(2, ti.i32, self.N_edges)

        # TODO
        # sparse matrix


        # init once and for all
        self.meshing()
        self.initialize()
        self.initialize_elements()
        self.updateLameCoeff()


    @ti.func
    def ij_2_index(self, i, j):
        return i * self.N_y + j
    
    # -----------------------meshing and init----------------------------
    @ti.kernel
    def meshing(self):
        # setting up triangles
        for i,j in ti.ndrange(self.N_x - 1, self.N_y - 1):
            # triangle id
            tid = (i * (self.N_y - 1) + j) * 2
            self.triangles[tid][0] = self.ij_2_index(i, j)
            self.triangles[tid][1] = self.ij_2_index(i + 1, j)
            self.triangles[tid][2] = self.ij_2_index(i, j + 1)

            tid = (i * (self.N_y - 1) + j) * 2 + 1
            self.triangles[tid][0] = self.ij_2_index(i, j + 1)
            self.triangles[tid][1] = self.ij_2_index(i + 1, j + 1)
            self.triangles[tid][2] = self.ij_2_index(i + 1, j)

        # setting up edges
        # edge id
        eid_base = 0

        # horizontal edges
        for i in range(self.N_x-1):
            for j in range(self.N_y):
                eid = eid_base+i*self.N_y+j
                self.edges[eid] = [self.ij_2_index(i, j), self.ij_2_index(i+1, j)]

        eid_base += (self.N_x-1)*self.N_y
        # vertical edges
        for i in range(self.N_x):
            for j in range(self.N_y-1):
                eid = eid_base+i*(self.N_y-1)+j
                self.edges[eid] = [self.ij_2_index(i, j), self.ij_2_index(i, j+1)]

        eid_base += self.N_x*(self.N_y-1)
        # diagonal edges
        for i in range(self.N_x-1):
            for j in range(self.N_y-1):
                eid = eid_base+i*(self.N_y-1)+j
                self.edges[eid] = [self.ij_2_index(i+1, j), self.ij_2_index(i, j+1)]

    @ti.kernel
    def initialize(self):
        self.YoungsModulus[None] = 1e6
        # init position and velocity
        for i, j in ti.ndrange(self.N_x, self.N_y):
            index = self.ij_2_index(i, j)
            self.x[index] = ti.Vector([self.init_x + i * self.dx, self.init_y + j * self.dx])
            self.v[index] = ti.Vector([0.0, 0.0])

    @ti.func
    def compute_D(self, i):
        a = self.triangles[i][0]
        b = self.triangles[i][1]
        c = self.triangles[i][2]
        return ti.Matrix.cols([self.x[b] - self.x[a], self.x[c] - self.x[a]])

    @ti.kernel
    def initialize_elements(self):
        for i in range(self.N_triangles):
            Dm = self.compute_D(i)
            self.elements_Dm_inv[i] = Dm.inverse()
            self.elements_V0[i] = ti.abs(Dm.determinant())/2
    
    # ----------------------core-----------------------------
    @ti.func
    def compute_R_2D(self, F):
        R, S = ti.polar_decompose(F, ti.f32)
        return R

    @ti.func
    def compute_gradient(self):
        # clear gradient
        for i in self.grad:
            self.grad[i] = ti.Vector([0, 0])

        # gradient of elastic potential
        for i in range(self.N_triangles):
            Ds = self.compute_D(i)
            F = Ds@self.elements_Dm_inv[i]
            # # co-rotated linear elasticity
            # R = compute_R_2D(F)
            Eye = ti.Matrix.cols([[1.0, 0.0], [0.0, 1.0]])
            # # first Piola-Kirchhoff tensor
            # P = 2*LameMu[None]*(F-R) +\
            #     LameLa[None]*((R.transpose())@F-Eye).trace()*R
            # stvk
            P = self.LameMu[None] * F @ (F.transpose() @ F - Eye) +\
                self.LameLa[None] * (0.5*(F.transpose() @ F - Eye)).trace() * F
            #assemble to gradient
            H = self.elements_V0[i] * P @ (self.elements_Dm_inv[i].transpose())
            a,b,c = self.triangles[i][0],self.triangles[i][1],self.triangles[i][2]
            gb = ti.Vector([H[0,0], H[1, 0]])
            gc = ti.Vector([H[0,1], H[1, 1]])
            ga = -gb-gc
            self.grad[a] += ga
            self.grad[b] += gb
            self.grad[c] += gc     
    
    @ti.func
    def clear_force(self):
        for i in self.force:
            self.force[i] = ti.Vector([0.0, 0.0])

    @ti.kernel
    def compute_force(self):
        self.clear_force()
        self.compute_gradient()
        for i in range(self.N):
            self.force[i] = (-self.grad[i] - ti.Vector([0.0, self.g])) * self.m

    @ti.kernel
    def compute_force_Jacobians(self):
        for i in self.spring:
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            pos1, pos2 = self.pos[idx1], self.pos[idx2]
            dx = pos1 - pos2
            I = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
            dxtdx = ti.Matrix([[dx[0] * dx[0], dx[0] * dx[1]],
                               [dx[1] * dx[0], dx[1] * dx[1]]])
            l = dx.norm()
            if l != 0.0:
                l = 1.0 / l
            self.Jx[i] = (I - self.rest_len[i] * l *
                          (I - dxtdx * l**2)) * self.ks

    @ti.kernel
    def update(self, h: ti.f32):
        # perform time integration
        for i in range(self.N):
            # symplectic integration
            # elastic force + gravitation force, divding mass to get the acceleration
            acc = -self.grad[i]/self.m - ti.Vector([0.0, self.g])
            self.v[i] += h*acc
            self.x[i] += h*self.v[i]

        # explicit damping (ether drag)
        for i in self.v:
            if damping_toggle[None]:
                self.v[i] *= ti.exp(-h*5)

        # enforce boundary condition
        for i in range(self.N):
            if picking[None]:           
                r = self.x[i]-curser[None]
                if r.norm() < curser_radius:
                    self.x[i] = curser[None]
                    self.v[i] = ti.Vector([0.0, 0.0])
                    pass

        for j in range(self.N_y):
            ind = self.ij_2_index(0, j)
            self.v[ind] = ti.Vector([0, 0])
            self.x[ind] = ti.Vector([self.init_x, self.init_y + j * self.dx])  # rest pose attached to the wall

        for i in range(self.N):
            if self.x[i][0] < self.init_x:
                self.x[i][0] = self.init_x
                self.v[i][0] = 0
    
    @ti.kernel
    def cgUpdatePosVel(self, h: ti.f32):
        for i in self.pos:
            self.vel[i] = self.x[i]
            self.pos[i] += h * self.vel[i]

    @ti.kernel
    def compute_RHS(self, h: ti.f32):
        #rhs = b = h * force + M @ v
        for i in range(self.NV):
            self.b[i] = h * self.force[i] + self.mass[i] * self.vel[i]

    @ti.func
    def dot(self, v1, v2):
        result = 0.0
        for i in range(self.NV):
            result += v1[i][0] * v2[i][0]
            result += v1[i][1] * v2[i][1]
        return result

    @ti.func
    def A_mult_x(self, h, dst, src):
        # A = M - h^2 * Jx 
        coeff = -h**2
        for i in range(self.NV):
            dst[i] = self.mass[i] * src[i]
        for i in range(self.NE):
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            temp = self.Jx[i] @ (src[idx1] - src[idx2])
            dst[idx1] -= coeff * temp
            dst[idx2] += coeff * temp
        # Attachment constraint
        Attachment1, Attachment2 = self.N, self.NV - 1
        dst[Attachment1] -= coeff * self.kf * src[Attachment1]
        dst[Attachment2] -= coeff * self.kf * src[Attachment2]

    # conjugate gradient solving

    @ti.kernel
    def before_ite(self) -> ti.f32:
        for i in range(self.NV):
            self.x[i] = ti.Vector([0.0, 0.0])
        self.A_mult_x(h, self.Ax, self.x)  # Ax = A @ x
        for i in range(self.NV):  # r = b - A @ x
            self.r[i] = self.b[i] - self.Ax[i]
        for i in range(self.NV):  # d = r
            self.d[i] = self.r[i]
        delta_new = self.dot(self.r, self.r)
        return delta_new

    @ti.kernel
    def run_iteration(self, delta_old: ti.f32) -> ti.f32:
        self.A_mult_x(h, self.Ad, self.d)  # Ad = A @ d
        alpha = delta_old / self.dot(self.d,
                                     self.Ad)  # alpha = (r^T * r) / dot(d, Ad)
        for i in range(self.NV):
            self.x[i] += alpha * self.d[i]  # x^{i+1} = x^{i} + alpha * d
            self.r[i] -= alpha * self.Ad[i]  # r^{i+1} = r^{i} + alpha * Ad
        delta_new = self.dot(self.r, self.r)
        beta = delta_new / delta_old
        for i in range(self.NV):
            self.d[i] = self.r[i] + beta * self.d[
                i]  #p^{i+1} = r^{i+1} + beta * p^{i}
        return delta_new

    def cg(self, h: ti.f32):
        delta_new = self.before_ite()
        ite, iteMax = 0, 2 * self.NV
        while ite < iteMax and delta_new > 1.0e-6:
            delta_new = self.run_iteration(delta_new)
            ite += 1
    
    def update_cg(self, h):
        self.compute_force()
        self.compute_force_Jacobians()
        self.compute_RHS(h)
        self.cg(h)
        self.cgUpdatePosVel(h)


    @ti.kernel
    def updateLameCoeff(self):
        E = self.YoungsModulus[None]
        nu = self.PoissonsRatio[None]
        self.LameLa[None] = E*nu / ((1+nu)*(1-2*nu))
        self.LameMu[None] = E / (2*(1+nu))
    
    def display(self, gui, radius=5, color=0xffffff):
        pos = self.x.to_numpy()
        for i in range(self.N_edges):
            a, b = self.edges[i][0], self.edges[i][1]
            gui.line((pos[a][0], pos[a][1]),
                    (pos[b][0], pos[b][1]),
                    radius=1,
                    color=0xFFFF00)
        gui.line((self.init_x, 0.0), (self.init_x, 1.0), color=0xFFFFFF, radius=4)


if __name__ == "__main__":
    ti.init(arch=ti.cpu)

    # global control
    paused = True
    damping_toggle = ti.field(ti.i32, ())
    curser = ti.Vector.field(2, ti.f32, ())
    picking = ti.field(ti.i32,())

    dx = 1/32
    curser_radius = dx/2
    beam = CrossBeam(N_x=20, N_y=4, dx=dx)
    gui = ti.GUI('Implicit FEM', res=(800, 800))
    pause = False
    h, max_step = 0.0001, 3
    while gui.running:
        picking[None] = 0
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                exit()
            elif e.key == 'r':
                beam.initialize()
                paused = True
            elif e.key == '0':
                beam.YoungsModulus[None] *= 1.1
            elif e.key == '9':
                beam.YoungsModulus[None] /= 1.1
                if beam.YoungsModulus[None] <= 0:
                    beam.YoungsModulus[None] = 0
            elif e.key == '8':
                beam.PoissonsRatio[None] = beam.PoissonsRatio[None]*0.9+0.05 # slowly converge to 0.5
                if beam.PoissonsRatio[None] >= 0.499:
                    beam.PoissonsRatio[None] = 0.499
            elif e.key == '7':
                beam.PoissonsRatio[None] = beam.PoissonsRatio[None]*1.1-0.05
                if beam.PoissonsRatio[None] <= 0:
                    beam.PoissonsRatio[None] = 0
            elif e.key == ti.GUI.SPACE:
                paused = not paused
            elif e.key =='d' or e.key == 'D':
                damping_toggle[None] = not damping_toggle[None]
            elif e.key =='p' or e.key == 'P': # step-forward
                for i in range(max_step):
                    beam.compute_gradient()
                    beam.update(h)
                    beam.update_cg(h)
            beam.updateLameCoeff()

        if gui.is_pressed(ti.GUI.LMB):
            curser[None] = gui.get_cursor_pos()
            picking[None] = 1

        # numerical time integration
        # if not paused:
        #     for i in range(substepping):
        #         if using_auto_diff:
        #             total_energy[None]=0
        #             with ti.Tape(total_energy):
        #                 compute_total_energy()
        #         else:
        #             compute_gradient()
        #         update()
        if not paused:
            for i in range(max_step):
                beam.compute_gradient()
                beam.update(h)
                beam.update_cg(h)
        beam.updateLameCoeff()

        beam.display(gui)
    
        if picking[None]:
            gui.circle((curser[None][0], curser[None][1]), radius=curser_radius*800, color=0xFF8888)

        # text
        gui.text(
            content=f'9/0: (-/+) Young\'s Modulus {beam.YoungsModulus[None]:.1f}', pos=(0.6, 0.9), color=0xFFFFFF)
        gui.text(
            content=f'7/8: (-/+) Poisson\'s Ratio {beam.PoissonsRatio[None]:.3f}', pos=(0.6, 0.875), color=0xFFFFFF)
        if damping_toggle[None]:
            gui.text(
                content='D: Damping On', pos=(0.6, 0.85), color=0xFFFFFF)
        else:
            gui.text(
                content='D: Damping Off', pos=(0.6, 0.85), color=0xFFFFFF)
        gui.show()
        