import numpy as np
from fenics import set_log_active
from fenics import IntervalMesh, FunctionSpace
from fenics import near, DirichletBC
from fenics import Expression, Constant
from fenics import Function, TestFunction, TrialFunction, interpolate
from fenics import inner, dx, grad
from fenics import lhs, rhs, solve


def solve_linear_pde(
    u_D_array,
    T,
    D=1,
    C1=0,
    num_r=100,
    min_r=0.001,
    tol=1e-14,
    degree=1,
):
    # disable logging
    set_log_active(False)

    num_t = len(u_D_array)
    dt = T / num_t  # time step size

    mesh = IntervalMesh(num_r, min_r, 1)
    r = mesh.coordinates().flatten()
    r_args = np.argsort(r)
    V = FunctionSpace(mesh, "P", 1)

    # Define boundary conditions
    # Dirichlet condition at R
    def boundary_at_R(x, on_boundary):
        return on_boundary and near(x[0], 1, tol)

    D = Constant(D)
    u_D = Constant(u_D_array[0])
    bc_at_R = DirichletBC(V, u_D, boundary_at_R)

    # Define initial values for free c
    c_0 = Expression("C1", C1=C1, degree=degree)
    c_n = interpolate(c_0, V)

    # Define variational problem
    c = TrialFunction(V)
    v = TestFunction(V)

    # define Constants
    r_squ = Expression("4*pi*pow(x[0],2)", degree=degree)

    F_tmp = (
        D * dt * inner(grad(c), grad(v)) * r_squ * dx
        + c * v * r_squ * dx
        - c_n * v * r_squ * dx
    )
    a, L = lhs(F_tmp), rhs(F_tmp)
    u = Function(V)

    data_c = np.zeros((num_t, len(r)), dtype=np.double)

    for n in range(num_t):
        u_D.assign(u_D_array[n])

        # Compute solution
        solve(a == L, u, bc_at_R)
        data_c[n, :] = u.vector().vec().array

        c_n.assign(u)

    data_c = data_c[:, r_args[::-1]]
    r = r[r_args]

    return data_c, r
