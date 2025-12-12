import numpy as np
from skfem import (
    Basis, ElementTriP2, ElementTriP1, ElementVector,
    asm, bmat, condense, solve, Mesh, BilinearForm
)
import pandas as pd
from skfem.helpers import grad, dot
from skfem.models.general import divergence
from skfem.models.poisson import vector_laplace


# cargar malla
malla = Mesh.load("data/Ugeom.obj", force_meshio_type='triangle')
nodes  = malla.p
facets = malla.facets
boundary_facets = malla.boundary_facets()
boundary_nodes = malla.boundary_nodes()

#etiquetar fronteras
pared   = []
salida  = []
entrada = []
for i,e in enumerate(boundary_facets):
    facets_e = facets[:, e]
    nodox = nodes[0, facets_e]
    nodoy = nodes[1, facets_e]
    m = np.array([np.mean(nodox), np.mean(nodoy)])

    if m[1] > -2.0:
        pared.append(e)
    if m[1] <= -2.0 and m[0] > 0.0:
        salida.append(e)
    if m[1] <= -2.0 and m[0] < 0.0:
        entrada.append(e)

pared = np.array(pared)
salida = np.array(salida)
entrada = np.array(entrada)
coordenadas = nodes[:, boundary_nodes]

# Definir elementos y bases (P2 para velocidad, P1 para presión)
element = {
    'u': ElementVector(ElementTriP2()),
    'p': ElementTriP1(),
}
basis = {
    'u': Basis(malla, element['u'], intorder=4),
    'p': Basis(malla, element['p'], intorder=4),
}
basis_u, basis_p = basis['u'], basis['p']
Nu, Np = basis_u.N, basis_p.N
N      = Nu + Np

# Ensamblaje de matrices
@BilinearForm
def convection(u, v, w):
    advection_field = w['w']
    grad_u = grad(u)
    return np.einsum('j...,ij...,i...->...', advection_field, grad_u, v)


A =  asm(vector_laplace, basis_u)               
B = -asm(divergence, basis_u, basis_p)   
F_u = basis['u'].zeros()
F_p = basis['p'].zeros()
F = np.hstack([F_u, F_p])

# Condiciones de borde
def u_in_y(x, y):
    return  (x-xin[1])*(xin[0]-x) + y*0.0

dofs_entrada  = basis_u.get_dofs(entrada).all()
dofs_salida   = basis_u.get_dofs(salida).all()
dofs_p_salida = basis_p.get_dofs(salida).all()

xout   = basis_u.doflocs[0, dofs_salida[::2]] 
yout   = basis_u.doflocs[1, dofs_salida[1::2]]
xin   = basis_u.doflocs[0, dofs_entrada[::2]] 
yin   = basis_u.doflocs[1, dofs_entrada[1::2]]

x_boundaries = np.zeros(Nu+Np)
x_boundaries[dofs_entrada[::2]]  = 0.0
x_boundaries[dofs_entrada[1::2]] = u_in_y(xin, yin)/ np.max(u_in_y(xin, yin)) 

x_boundaries[Nu + dofs_p_salida[0]] = 0.0

D_all = np.concatenate([
    basis_u.get_dofs(pared).all(),
    basis_u.get_dofs(entrada).all(),
    Nu+np.array([dofs_p_salida[0]])
]) 

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
        sol = solve(*condense(K, F, D=D_all, x=x_boundaries))
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

# Calcular solución inicial de stokes
mu = 0.035
K_stokes = bmat([[mu * A,     B.T],  
                 [B,         None]], format='csr')

sol0 = solve(*condense(K_stokes, F, D=D_all, x=x_boundaries))
u_ref = sol0[:Nu].copy()
p_ref = sol0[Nu:Nu+Np].copy()

# Resolver incrementanto Re
Re = 100
Re_linspace = np.linspace(1, Re, 50)

for R in Re_linspace:
    print(f"Resolviendo para Re = {R:.2f}")
    u_ref, p_ref, flag = solve_ns_picard(u_ref, p_ref, R, max_iter=1000, tol=1e-13)

    if not flag:
        print(f"No se pudo converger para este Re = {R:.2f}.")
        break

u_sol = u_ref
p_sol = p_ref

u_scale = 3
p_scale = 1.05 * u_scale**2

# Exportar soluciones de referencia como csv
df_u = pd.DataFrame({
    'u_sol': u_sol*u_scale,
})
df_p = pd.DataFrame({
    'p_sol': p_sol*p_scale,
})
df_u.to_csv("data/reference_solution_velocity_navierstokes.csv", index=False)
df_p.to_csv("data/reference_solution_pressure_navierstokes.csv", index=False)