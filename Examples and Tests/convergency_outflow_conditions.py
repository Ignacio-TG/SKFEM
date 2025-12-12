import numpy as np
from skfem import (
    MeshTri, Basis, FacetBasis, ElementTriP2, ElementTriP1, ElementVector,
    asm, bmat, condense, solve, BilinearForm, LinearForm, Mesh, Functional
)
from skfem.models.general import divergence
from skfem.models.poisson import vector_laplace
from skfem.helpers import grad, dot, laplacian
from skfem.helpers import laplacian, precompute_operators
import matplotlib.pyplot as plt

import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, eigs, splu, LinearOperator

mu = 1
Re = 1
def u_func_x(x, y):
    return (np.cos(np.pi * x) -1 )/ np.pi
def u_func_y(x, y):
    return np.sin(np.pi * x)*y
def p_exact(x, y):
    return np.sin(np.pi * x)

def f_exact_x_stokes(x,y):
    return np.pi*np.cos(np.pi*x) + np.pi*np.cos((np.pi*x))

def f_exact_y_stokes(x,y):
    return np.pi**2 * np.sin(np.pi*x)*y


@BilinearForm
def mass_matrix(u, v, w):
    return dot(u, v)

@LinearForm
def rhs_u(v, w):
    x, y = w.x
    fx = f_exact_x_stokes(x, y)
    fy = f_exact_y_stokes(x, y)
    return fx * v[0] + fy * v[1]

@Functional
def traction_u_normal(w):
    u  = w['u']
    gu = u.grad   
    t  = dot(gu, w.n)
    return dot(t, w.n)



@Functional
def traction_p_normal(w):
    p = w['p']
    # -p * (n·n); en 'outflow' n es unitario, luego n·n = 1
    return -p

N_elem = [8, 16, 32, 64, 128, 256]
errors = []

for n_i in N_elem:
    # Definir dominio y mallado
    mesh = MeshTri.init_tensor(
        np.linspace(0.0, 1.0, n_i + 1), 
        np.linspace(0.0, 1.0, n_i + 1)
    )

    # Asignar ID a las fronteras
    mesh = mesh.with_boundaries({
        'wall':   lambda x: x[0] == 0.0,
        'inflow': lambda x: (x[0] == 1.0) | (x[1] == 0.0),
        'outflow': lambda x: x[1] == 1.0,
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


    # Ensamblaje de matrices
    A =  asm(vector_laplace, basis_u)               
    B = -asm(divergence, basis_u, basis_p)   

    x_boundary = np.zeros(Nu + Np)

    # Fronteras de la velocidad

    # Obtener DOFs de las fronteras de la base de velocidad
    dofs_wall     = basis_u.get_dofs('wall').all()
    dofs_inflow   = basis_u.get_dofs('inflow').all()
    dofs_outflow  = basis_u.get_dofs('outflow').all()

    # Elementos con condiciones de Dirichlet para la velocidad
    # Impares y, pares x
    y_wall    = basis_u.doflocs[1, dofs_wall[1::2]] 
    y_inflow  = basis_u.doflocs[1, dofs_inflow[1::2]]
    y_outflow = basis_u.doflocs[1, dofs_outflow[1::2]]

    x_wall   = basis_u.doflocs[0, dofs_wall[::2]]  
    x_inflow = basis_u.doflocs[0, dofs_inflow[::2]]
    x_outflow    = basis_u.doflocs[0, dofs_outflow[::2]]

    x_boundary[dofs_wall[::2]]    =  u_func_x(x_wall, y_wall)
    x_boundary[dofs_wall[1::2]]   =  u_func_y(x_wall, y_wall)

    x_boundary[dofs_inflow[::2]]   =  u_func_x(x_inflow, y_inflow)
    x_boundary[dofs_inflow[1::2]]  =  u_func_y(x_inflow, y_inflow)

    # x_boundary[dofs_outflow[::2]]     =  u_func_x(x_outflow, y_outflow)  # componente x
    # x_boundary[dofs_outflow[1::2]]    =  u_func_y(x_outflow, y_outflow)  # componente y


    dofs_u_boundary = np.concatenate([
        dofs_wall,
        dofs_inflow,
        # dofs_outflow
    ])

    x_boundary[Nu + 0]  = p_exact(0.0, 0.0)
    dofs_p_boundary = np.array([Nu+0])# + np.concatenate([np.array([0])])

    D_all = np.unique(np.concatenate([dofs_u_boundary, dofs_p_boundary]))



    b_u = asm(rhs_u, basis_u) 
    b_p = np.zeros(Np)  
    F   = np.concatenate([b_u, b_p])

    mu = 1.0

    # Calcular solución inicial de stokes
    K_stokes = bmat([[mu * A,     B.T],  
                    [B,         None]], format='csr')

    sol0 = solve(*condense(K_stokes, F, D=D_all, x=x_boundary))
    u_ref = sol0[:Nu].copy()
    p_ref = sol0[Nu:Nu+Np].copy()

    # Valores de u_h en puntos de cuadratura de la frontera superior
    fbasis_u = basis_u.boundary('outflow', intorder=4)
    u_b      = fbasis_u.interpolate(u_ref)  # shape (2, nqp)
    fbasis_p = basis_p.boundary('outflow', intorder=4)
    p_b      = fbasis_p.interpolate(p_ref)

    T_u = traction_u_normal.assemble(fbasis_u, u=u_b)

    T_p = traction_p_normal.assemble(fbasis_p, p=p_b)

    T_total = T_u + T_p

    errors.append(np.abs(T_total))  
    print("Integral de traccion normal (FEM):", T_total)


# Exportar resultados
import pandas as pd
df = pd.DataFrame({
    'N_elem': N_elem,
    'Error_traccion_normal': errors
})

df.to_csv('convergency_outflow_conditions.csv', index=False)