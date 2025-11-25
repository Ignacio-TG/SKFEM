import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skfem import (
    MeshTri, Basis, FacetBasis, ElementTriP2, ElementTriP1, ElementVector,
    asm, bmat, condense, solve, BilinearForm, LinearForm
)
from skfem.models.general import divergence
from skfem.models.poisson import vector_laplace, mass
from skfem.helpers import grad, dot, laplacian

import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, eigs, splu, LinearOperator

import scipy.linalg


def build_problem_stokes(problem, N):

    if problem == 1:
        # Definir dominio y mallado
        mesh = MeshTri.init_tensor(
            np.linspace(-1.0, 1.0, N + 1), 
            np.linspace(-1.0, 1.0, N + 1)
        )

        mesh = mesh.with_boundaries({
            'left':   lambda x: x[0] == -1.0,
            'right':  lambda x: x[0] == 1.0,
            'bottom': lambda x: x[1] == -1.0,
            'top':    lambda x: x[1] == 1.0,
        })
    else:
        # Definir dominio y mallado
        mesh = MeshTri.init_tensor(
            np.linspace(0.0, 1.0, N + 1), 
            np.linspace(0.0, 1.0, N + 1)
        )

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
    N = Nu + Np

    @BilinearForm
    def mass_matrix(u, v, w):
        return dot(u, v)

    # Ensamblaje de matrices
    A =  asm(vector_laplace, basis_u)               
    B = -asm(divergence, basis_u, basis_p)   
    M =  asm(mass_matrix, basis_u)    

    # Construcción del ssitema
    K = bmat([[A,    B.T],
            [B,    None]], format='csr') 

    zeros = sp.csr_matrix((basis_p.N, basis_p.N))  
    L = bmat([[M,   None],
            [None, zeros]], format='csr')


    if problem < 3:
        dofs_left     = basis_u.get_dofs('left').all()
        dofs_right    = basis_u.get_dofs('right').all()
        dofs_top      = basis_u.get_dofs('top').all()
        dofs_bottom   = basis_u.get_dofs('bottom').all()

        D_all = np.unique(np.concatenate([
            dofs_left,
            dofs_right,
            dofs_top,
            dofs_bottom,
            Nu + np.array([0])
        ]) )

    else:
        # dofs_left     = basis_u.get_dofs('left').all()
        # dofs_right    = basis_u.get_dofs('right').all()
        # dofs_top      = basis_u.get_dofs('top').all()
        dofs_bottom   = basis_u.get_dofs('bottom').all()

        D_all = np.unique(np.concatenate([
            # dofs_left,
            # dofs_right,
            # dofs_top,
            dofs_bottom,
            Nu + np.array([0])
        ]) )
    
    A_sys, xI, I   = condense(K, D=D_all)
    M_sys, xIM, IM = condense(L, D=D_all)
    vals, vecs     = eigs(A_sys, k=10, M=M_sys, sigma=0.0, which='LM', OPpart='r', tol=1e-12)

    return vals, vecs

#%%  
if __name__ == "__main__":
    N = [4, 8, 16, 32, 64, 128]
    problem = [1, 2, 3]

    vals_1 = []
    vals_2 = []
    vals_3 = []

    for n in N:
        for p in problem:
            vals, vecs = build_problem_stokes(p, n)
            if p == 1:
                vals_1.append(np.real(vals).tolist())
            elif p == 2:
                vals_2.append(np.real(vals).tolist())
            else:
                vals_3.append(np.real(vals).tolist())
            print(f'Problema {p}, N={n}, listo!')
    # Exportar resultados a tablas
    # Problema 1
    df_1 = pd.DataFrame(vals_1, columns=[f'lambda_{i+1}' for i in range(len(vals_1[0]))])
    df_1.insert(0, 'N', N)
    df_1.to_csv('data/eigenvalues_problem1.csv', index=False)
    # Problema 2
    df_2 = pd.DataFrame(vals_2, columns=[f'lambda_{i+1}' for i in range(len(vals_2[0]))])
    df_2.insert(0, 'N', N)
    df_2.to_csv('data/eigenvalues_problem2.csv', index=False)
    # Problema 3
    df_3 = pd.DataFrame(vals_3, columns=[f'lambda_{i+1}' for i in range(len(vals_3[0]))])
    df_3.insert(0, 'N', N)
    df_3.to_csv('data/eigenvalues_problem3.csv', index=False)
    
# %%
