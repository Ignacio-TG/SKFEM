import numpy as np
from skfem import (
    MeshTri, Basis, FacetBasis, ElementTriP2, ElementTriP1, ElementVector,
    asm, bmat, condense, solve, BilinearForm, LinearForm, Mesh, Functional
)
from skfem.models.general import divergence
from skfem.models.poisson import vector_laplace
from skfem.helpers import grad, dot
import pandas as pd

# Definir dominio y mallado
nx, ny = 32, 32
mesh = MeshTri.init_tensor(
    np.linspace(0.0, 1.0, nx + 1), 
    np.linspace(0.0, 1.0, ny + 1)
)

# Asignar ID a las fronteras
mesh = mesh.with_boundaries({
    'left':   lambda x: x[0] == 0.0,
    'right':  lambda x: x[0] == 1.0,
    'bottom': lambda x: x[1] == 0.0,
    'top':    lambda x: x[1] == 1.0,
})

# Definir elementos y bases (P2 para velocidad, P1 para presión)
element = {
    'u': ElementVector(ElementTriP2()),
    'p': ElementTriP1(),
}
basis = {
    'u': Basis(mesh, element['u'], intorder=4),
    'p': Basis(mesh, element['p'], intorder=4),
}
basis_u, basis_p = basis['u'], basis['p']
Nu, Np = basis_u.N, basis_p.N
N      = Nu + Np

@BilinearForm
def mass_matrix(u, v, w):
    return dot(u, v)

# Término convectivo linealizado
@BilinearForm
def convection(u, v, w):
    advection_field = w['w']
    grad_u = grad(u)

    return np.einsum('j...,ij...,i...->...', advection_field, grad_u, v)

# Ensamblaje de matrices
A =  asm(vector_laplace, basis_u)               
B = -asm(divergence, basis_u, basis_p)   

mu = 1
Re = 10
def u_func_x(x, y):
    return np.sin(np.pi * y) + 0*x
def u_func_y(x, y):
    return np.sin(np.pi * x) + 0*y
def p_exact(x, y):
    return np.exp(x-y)

def f_exact_x_stokes(x,y):
    return np.exp(x-y) + np.pi**2 *mu * (np.sin(np.pi*y))

def f_exact_y_stokes(x,y):
    return -np.exp(x-y) + np.pi**2 *mu * (np.sin(np.pi*x))

def f_exact_x_ns(x,y):
    return  np.exp(x-y) + np.pi**2 * (np.sin(np.pi*y))/Re + np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)

def f_exact_y_ns(x,y):
    return -np.exp(x-y) + np.pi**2 * (np.sin(np.pi*x))/Re + np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)

x_boundary = np.zeros(Nu + Np)

# Fronteras de la velocidad

# Obtener DOFs de las fronteras de la base de velocidad
dofs_left     = basis_u.get_dofs('left').all()
dofs_right    = basis_u.get_dofs('right').all()
dofs_top      = basis_u.get_dofs('top').all()
dofs_bottom   = basis_u.get_dofs('bottom').all()

# Elementos con condiciones de Dirichlet para la velocidad
# Impares y, pares x
y_left   = basis_u.doflocs[1, dofs_left[1::2]] 
y_right  = basis_u.doflocs[1, dofs_right[1::2]]

x_left   = basis_u.doflocs[0, dofs_left[::2]]  
x_right  = basis_u.doflocs[0, dofs_right[::2]]

x_top    = basis_u.doflocs[0, dofs_top[::2]]
x_bottom = basis_u.doflocs[0, dofs_bottom[::2]]

y_top    = basis_u.doflocs[1, dofs_top[1::2]]
y_bottom = basis_u.doflocs[1, dofs_bottom[1::2]]


x_boundary[dofs_left[::2]]    =  u_func_x(x_left, y_left)
x_boundary[dofs_left[1::2]]   =  u_func_y(x_left, y_left)

x_boundary[dofs_right[::2]]   =  u_func_x(x_right, y_right)
x_boundary[dofs_right[1::2]]  =  u_func_y(x_right, y_right)

x_boundary[dofs_top[::2]]     =  u_func_x(x_top, y_top)  # componente x
x_boundary[dofs_top[1::2]]    =  u_func_y(x_top, y_top)  # componente y

x_boundary[dofs_bottom[::2]]  =  u_func_x(x_bottom, y_bottom)  # componente x
x_boundary[dofs_bottom[1::2]] =  u_func_y(x_bottom, y_bottom)  # componente y

dofs_u_boundary = np.concatenate([
    dofs_left,
    dofs_right,
    dofs_top,
    dofs_bottom
])

x_boundary[Nu + 0]  = p_exact(0.0, 0.0)
dofs_p_boundary = np.array([Nu+0])# + np.concatenate([np.array([0])])

D_all = np.unique(np.concatenate([dofs_u_boundary, dofs_p_boundary]))

@LinearForm
def rhs_u(v, w):
    x, y = w.x
    fx = f_exact_x_stokes(x, y)
    fy = f_exact_y_stokes(x, y)
    return fx * v[0] + fy * v[1]

b_u = asm(rhs_u, basis_u) 
b_p = np.zeros(Np)  
F   = np.concatenate([b_u, b_p])


# Calcular solución inicial de stokes
K_stokes = bmat([[mu * A,     B.T],  
                 [B,         None]], format='csr')

sol0 = solve(*condense(K_stokes, F, D=D_all, x=x_boundary))
u_ref = sol0[:Nu].copy()
p_ref = sol0[Nu:Nu+Np].copy()

@LinearForm
def rhs_u(v, w):
    x, y = w.x
    fx = f_exact_x_ns(x, y)
    fy = f_exact_y_ns(x, y)
    return fx * v[0] + fy * v[1]

b_u = asm(rhs_u, basis_u) 
b_p = np.zeros(Np)  
F   = np.concatenate([b_u, b_p])

def solve_ns_picard(u_init, p_init, Re, max_iter, tol):
    u = u_init
    p = p_init
    for it in range(max_iter):
        # Campo de advección congelado w := u^(it) en puntos de cuadratura
        W = basis_u.interpolate(u)   

        # Ensambla bloque convectivo C(w)
        C = asm(convection, basis_u, w=W)

        # Matriz bloque del paso linealizado
        K = bmat([[(1/Re) * A + C, B.T ],
                  [B,              None]], format='csr')

        # Resolver
        sol = solve(*condense(K, F, D=D_all, x=x_boundary))
        u_new = sol[:Nu]
        p_new = sol[Nu:Nu+Np]

        # Criterio de convergencia
        du = u_new - u
        rel_u = np.linalg.norm(du) / (np.linalg.norm(u_new) + 1e-16)

        dp = p_new - p
        rel_p = np.linalg.norm(dp) / (np.linalg.norm(p_new) + 1e-16)

        # # Sub-relajación si se desea
        u = u_new
        p = p_new

        if rel_u < tol and rel_p < tol:
            print(f"Convergió en {it+1} iteraciones, residuo {max(rel_u, rel_p):.4e}")
            return u, p, True
    print("No convergió en el número máximo de iteraciones")
    return u, p, False

# Resolver incrementanto Re
Re = 10
Re_linspace = np.linspace(1, Re, 10)

for R in Re_linspace:
    print(f"Resolviendo para Re = {R:.2f}")
    u_ref, p_ref, flag = solve_ns_picard(u_ref, p_ref, R, max_iter=500, tol=1e-12)

    if not flag:
        print(f"No se pudo converger para este Re = {R:.2f}.")
        break

u_sol = u_ref
p_sol = p_ref

# Exportar soluciones de referencia como csv
df_u = pd.DataFrame({
    'u_sol': u_sol,
})
df_p = pd.DataFrame({
    'p_sol': p_sol,
})
df_u.to_csv("data/reference_solution_velocity_navierstokes_sincos.csv", index=False)
df_p.to_csv("data/reference_solution_pressure_navierstokes_sincos.csv", index=False)