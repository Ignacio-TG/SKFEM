import numpy as np
from skfem import (
    MeshTri, Basis, FacetBasis, ElementTriP2, ElementTriP1, ElementVector,
    asm, bmat, condense, solve, BilinearForm, LinearForm, Mesh, ElementTetP2, ElementTetP1
)
from skfem.utils import  solver_iter_krylov
from skfem.models.general import divergence
from skfem.models.poisson import vector_laplace
from skfem.helpers import grad, dot, laplacian
from skfem.helpers import laplacian, precompute_operators
import matplotlib.pyplot as plt

import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, eigs, splu, LinearOperator

import scipy.linalg
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import plotly.graph_objects as go
from skfem.mesh import MeshTet # MeshTet para mallas tetraédricas (3D)

mesh = Mesh.load("tubo.msh");

def get_boundary_facets_indices(mesh): 
    boundary_facets_idx = mesh.boundary_facets()
    facets  = mesh.facets
    nodes   = mesh.p
    
    pared_idx   = []
    salida_idx  = []
    entrada_idx = []

    for e in boundary_facets_idx:
        coords_facets = nodes[:, facets[:, e]]
        m = np.mean(coords_facets, axis=1) # m = [mean(x), mean(y), mean(z)]

        # Lógica de Etiquetado (usando Y = m[1] como eje vertical, cerca de -2)
        
        # Pared: Partes que no son la entrada/salida (e.g., Y > -2.0)
        if m[1] > -2.0:
            pared_idx.append(e)
            
        # Salida: Extremo derecho (Y en -2.0 y X > 0)
        elif m[1]==-2.0 and m[0] > 0.0:
            salida_idx.append(e)
            
        # Entrada: Extremo izquierdo (Y en -2.0 y X < 0)
        elif m[1]==-2.0 and m[0] < 0.0:
            entrada_idx.append(e)
            
    return np.array(pared_idx), np.array(salida_idx), np.array(entrada_idx)

wall, outflow, inflow = get_boundary_facets_indices(mesh)

# Definir elementos y bases (P2 para velocidad, P1 para presión) en 3D
element = {
    'u': ElementVector(ElementTetP2(), dim=3),
    'p': ElementTetP1(),
}
basis = {
    'u': Basis(mesh, element['u'], intorder=4),
    'p': Basis(mesh, element['p'], intorder=4),
}
basis_u, basis_p = basis['u'], basis['p']
Nu, Np = basis_u.N, basis_p.N
N      = Nu + Np

mu    = 0.035 # viscosidad dinámica
rho   = 1.0   # densidad
nu    = mu / rho # Viscosidad cinemática
V_car = 1.0    # Velocidad característica
L_car = 1.0    # Longitud característica
Re    = V_car * L_car / nu

@BilinearForm
def mass_matrix(u, v, w):
    return dot(u, v)


# Ensamblaje de matrices
A =  asm(vector_laplace, basis_u)               
B = -asm(divergence, basis_u, basis_p)   

F_u = basis['u'].zeros()
F_p = basis['p'].zeros()
F = np.hstack([F_u, F_p])

# DOFs de las fronteras
dofs_wall   = basis_u.get_dofs(wall).all()
dofs_outflow  = basis_u.get_dofs(outflow).all()
dofs_inflow = basis_u.get_dofs(inflow).all()

dofs_inflow_x = dofs_inflow[dofs_inflow % 3 == 0]
dofs_inflow_y = dofs_inflow[dofs_inflow % 3 == 1]
dofs_inflow_z = dofs_inflow[dofs_inflow % 3 == 2]
dofs_p_salida = basis_p.get_dofs(outflow).all()

xin = basis_u.doflocs[0, dofs_inflow_x]
yin = basis_u.doflocs[1, dofs_inflow_y]
zin = basis_u.doflocs[2, dofs_inflow_z]

# Inflow function
def inflow_velocity(x, y, z):
    x_min = np.min(xin)
    x_max = np.max(xin)
    z_min = np.min(zin)
    z_max = np.max(zin)
    # Velocity components: flow in y-direction
    return (x - x_min) * (x_max - x) * (z - z_min) * (z_max - z)

v_inflow = inflow_velocity(xin, yin, zin)

x_boundaries = np.zeros(Nu+Np)

# Inflow
x_boundaries[dofs_inflow_x] = 0.0
x_boundaries[dofs_inflow_y] = v_inflow/np.max(v_inflow)
x_boundaries[dofs_inflow_z] = 0.0

# Presión a la salida
x_boundaries[Nu + dofs_p_salida[0]] = 0.0

D_all = np.concatenate([
    basis_u.get_dofs(wall).all(),
    basis_u.get_dofs(inflow).all(),
    Nu+np.array([dofs_p_salida[0]])
]) 

# Término convectivo linealizado
@BilinearForm
def convection(u, v, w):
    advection_field = w['w']
    grad_u = grad(u)
    return np.einsum('j...,ij...,i...->...', advection_field, grad_u, v)

@BilinearForm
def convection2(u, v, w):
    advection_field = u
    grad_w = grad(w['w'])
    return np.einsum('j...,ij...,i...->...', advection_field, grad_w, v)

def solve_ns_newton(u_init, p_init, Re, max_iter, tol):
    u = u_init
    p = p_init
    for it in range(max_iter):
        # Campo de advección congelado w := u^(it) en puntos de cuadratura
        W = basis_u.interpolate(u)   

        # Ensambla bloque convectivo C(w)
        C2 = asm(convection, basis_u, w=W)

        # Ensambla derivada del bloque convectivo C'(w)
        C1 = asm(convection2, basis_u, w=W)

        # Matriz bloque del paso linealizado
        DF = bmat([[(1/Re) * A + C1 + C2, B.T ],
                  [B,                        None]], format='csr')

        F1 = C2*u + (1/Re)*A*u + B.T*p 
        F2 = B*u
        F_real = np.concatenate([F1, F2])

        # Resolver
        delta = solve(*condense(DF, F_real, D=D_all, x=x_boundaries*0))
        u_new = u - delta[:Nu]
        p_new = p - delta[Nu:Nu+Np]

        # Criterio de convergencia
        du = u_new - u
        rel_u = np.linalg.norm(du) / (np.linalg.norm(u_new) + 1e-16)

        dp = p_new - p
        rel_p = np.linalg.norm(dp) / (np.linalg.norm(p_new) + 1e-16)

        # # Sub-relajación si se desea
        u = u_new
        p = p_new

        # print(f"Iteración {it+1}: rel_u = {rel_u:.4e}, rel_p = {rel_p:.4e}")

        if rel_u < tol and rel_p < tol:
            print(f"Convergió en {it+1} iteraciones, residuo {max(rel_u, rel_p):.4e}")
            return u, p, True
    print("No convergió en el número máximo de iteraciones")
    return u, p, False


mu = 1.0

# Calcular solución inicial de stokes
K_stokes = bmat([[mu * A,     B.T],  
                 [B,         None]], format='csr')

sol0 = solve(*condense(K_stokes, F, D=D_all, x=x_boundaries))
u_ref = sol0[:Nu]
p_ref = sol0[Nu:Nu+Np]

# Resolver incrementanto Re
Re = 100
Re_linspace = np.linspace(10, Re, 2)

for R in Re_linspace:
    print(f"Resolviendo para Re = {R:.2f}")
    u_ref, p_ref, flag = solve_ns_newton(u_ref, p_ref, R, max_iter=500, tol=1e-12)

    if not flag:
        print(f"No se pudo converger para este Re = {R:.2f}.")
        break

u_sol = u_ref
p_sol = p_ref


np.savez_compressed("solucion_ns_3D.npz", u=u_sol, p=p_sol)
print('Done!')