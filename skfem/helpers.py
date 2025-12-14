"""Helper functions for defining forms."""

from typing import Union, Optional
import numpy as np
from numpy import ndarray, zeros_like
from skfem import Basis
from skfem.element import DiscreteField
from skfem.assembly.form.form import FormExtraParams


FieldOrArray = Union[DiscreteField, ndarray]


def jump(w: FormExtraParams, *args):
    if not hasattr(w, 'idx'):
        return args
    out = []
    for i, arg in enumerate(args):
        out.append((-1.) ** w.idx[i] * arg)
    return out[0] if len(out) == 1 else tuple(out)


def grad(u: DiscreteField):
    """Gradient."""
    return u.grad


def div(u: DiscreteField):
    """Divergence."""
    if u.div is not None:
        return u.div
    elif u.grad is not None:
        try:
            return np.einsum('ii...', u.grad)
        except ValueError:  # one-dimensional u?
            return u.grad[0]
    raise NotImplementedError


def curl(u: DiscreteField):
    """Curl."""
    if u.curl is not None:
        return u.curl
    elif u.grad is not None:
        if len(u.grad.shape) == 3 and u.grad.shape[0] == 2:
            # curl of scalar field
            return np.array([u.grad[1], -u.grad[0]])
        elif len(u.grad.shape) == 4 and u.grad.shape[0] == 2:
            # 2d-curl of vector field
            return u.grad[1, 0] - u.grad[0, 1]
        elif len(u.grad.shape) == 4 and u.grad.shape[0] == 3:
            # full 3d curl
            return np.array([
                u.grad[2, 1] - u.grad[1, 2],
                u.grad[0, 2] - u.grad[2, 0],
                u.grad[1, 0] - u.grad[0, 1],
            ])
    raise NotImplementedError


def d(u: DiscreteField):
    """Gradient, divergence or curl."""
    if u.grad is not None:
        return u.grad
    elif u.div is not None:
        return u.div
    elif u.curl is not None:
        return u.curl
    raise NotImplementedError


def sym_grad(u: DiscreteField):
    """Symmetric gradient."""
    return .5 * (u.grad + transpose(u.grad))


def dd(u: DiscreteField):
    """Hessian (for :class:`~skfem.element.ElementGlobal`)."""
    return u.hess


def ddd(u: DiscreteField):
    """Third derivative (for :class:`~skfem.element.ElementGlobal`)."""
    return u.grad3


def dddd(u: DiscreteField):
    """Fourth derivative (for :class:`~skfem.element.ElementGlobal`)."""
    return u.grad4


def inner(u: FieldOrArray, v: FieldOrArray):
    """Inner product between any matching tensors."""
    if isinstance(u, tuple) and isinstance(v, tuple):
        # support for ElementComposite
        out = []
        for i in range(len(v)):
            out.append(inner(u[i], v[i]))
        return sum(out)
    if len(u.shape) == 2:
        return u * v
    elif len(u.shape) == 3:
        return dot(u, v)
    elif len(u.shape) == 4:
        return ddot(u, v)
    raise NotImplementedError


def dot(u: FieldOrArray, v: FieldOrArray):
    """Dot product."""
    return np.einsum('i...,i...', u, v)


def ddot(u: FieldOrArray, v: FieldOrArray):
    """Double dot product."""
    return np.einsum('ij...,ij...', u, v)


def dddot(u: FieldOrArray, v: FieldOrArray):
    """Triple dot product."""
    return np.einsum('ijk...,ijk...', u, v)


def prod(u: FieldOrArray,
         v: FieldOrArray,
         w: Optional[FieldOrArray] = None):
    """Tensor product."""
    if w is None:
        return np.einsum('i...,j...->ij...', u, v)
    return np.einsum('i...,j...,k...->ijk...', u, v, w)


def mul(A: FieldOrArray, x: FieldOrArray):
    """Matrix multiplication."""
    return np.einsum('ij...,j...->i...', A, x)


def trace(T):
    """Trace of matrix."""
    return np.einsum('ii...', T)


def transpose(T):
    """Transpose of matrix."""
    return np.einsum('ij...->ji...', T)


def eye(w, n):
    """Create diagonal matrix with w on diagonal."""
    return np.array([[w if i == j else 0. * w for i in range(n)]
                     for j in range(n)])


def identity(w, N=None):
    """Create identity matrix."""
    if N is None:
        if len(w.shape) > 2:
            N = w.shape[-3]
        else:
            raise ValueError("Cannot deduce the size of the identity matrix. "
                             "Give an explicit keyword argument N.")
    return eye(np.ones(w.shape[-2:]), N)


def det(A):
    """Determinant of an array `A` over trailing axis (if any)."""
    detA = zeros_like(A[0, 0])
    if A.shape[0] == 3:
        detA = A[0, 0] * (A[1, 1] * A[2, 2] -
                          A[1, 2] * A[2, 1]) -\
               A[0, 1] * (A[1, 0] * A[2, 2] -
                          A[1, 2] * A[2, 0]) +\
               A[0, 2] * (A[1, 0] * A[2, 1] -
                          A[1, 1] * A[2, 0])
    elif A.shape[0] == 2:
        detA = A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1]
    return detA


def inv(A):
    """Inverse of an array `A` over trailing axis (if any)."""
    invA = zeros_like(A)
    detA = det(A)
    if A.shape[0] == 3:
        invA[0, 0] = (-A[1, 2] * A[2, 1] +
                      A[1, 1] * A[2, 2]) / detA
        invA[1, 0] = (A[1, 2] * A[2, 0] -
                      A[1, 0] * A[2, 2]) / detA
        invA[2, 0] = (-A[1, 1] * A[2, 0] +
                      A[1, 0] * A[2, 1]) / detA
        invA[0, 1] = (A[0, 2] * A[2, 1] -
                      A[0, 1] * A[2, 2]) / detA
        invA[1, 1] = (-A[0, 2] * A[2, 0] +
                      A[0, 0] * A[2, 2]) / detA
        invA[2, 1] = (A[0, 1] * A[2, 0] -
                      A[0, 0] * A[2, 1]) / detA
        invA[0, 2] = (-A[0, 2] * A[1, 1] +
                      A[0, 1] * A[1, 2]) / detA
        invA[1, 2] = (A[0, 2] * A[1, 0] -
                      A[0, 0] * A[1, 2]) / detA
        invA[2, 2] = (-A[0, 1] * A[1, 0] +
                      A[0, 0] * A[1, 1]) / detA
    elif A.shape[0] == 2:
        invA[0, 0] = A[1, 1] / detA
        invA[0, 1] = -A[0, 1] / detA
        invA[1, 0] = -A[1, 0] / detA
        invA[1, 1] = A[0, 0] / detA
    return invA


def cross(A, B):
    if A.shape[0] == 2:
        return A[0] * B[1] - A[1] * B[0]
    if A.shape[0] == 3:
        return np.array([
            A[1] * B[2] - A[2] * B[1],
            A[2] * B[0] - A[0] * B[2],
            A[0] * B[1] - A[1] * B[0]
        ])

def laplacian(u: np.ndarray, basis):
    """
    Compute the element-wise constant Laplacian of a P2 finite element solution with an affine mapping.

    ## Inputs
    - u : numpy.ndarray
        Vector of nodal coefficients (finite element solution), shape (n_dofs,).
    - basis : skfem.Basis
        Basis associated to Tri P2 elements (or, more generally, affine P2 elements).
        The basis is expected to provide mapping().invA of shape (dim, dim, nt) and
        element local basis routines compatible with P2.
    
    ## Output
    - lap_uh : numpy.ndarray
        Array of element-wise constant values of Δu_h, shape (nt,), where nt is the
        number of elements.
    
    Notes
    -----
    - The implementation assumes an affine mapping per element so that the Hessian of
      reference basis functions is constant on each physical element.

    """

    # Extract affine mapping inverse Jacobians
    invA = basis.mesh.mapping().invA                 # (dim, dim, nt)
    G    = np.einsum('aik,bik->abk', invA, invA)     # (dim, dim, nt)
    dim  = invA.shape[0]                             # 2 for 2D, 3 for 3D

    # Set reference point based on dimension
    if dim == 2:
        Xref = (1/3, 1/3)  # Triangle center
    elif dim == 3:
        Xref = (1/4, 1/4, 1/4)  # Tetrahedron center

    
    # Reference Hessians of local basis functions
    nlb  = basis.elem.doflocs.shape[0]     # N° local basis (P2 -> 6)
    Hhat = np.zeros((dim, dim, nlb))
    
    for i in range(nlb):
        _, _, H = basis.elem.lbasis(Xref, i)    
        Hhat[:, :, i] = H

    # Calculate laplacian of reference bases mapped to physical element
    laps = np.einsum('abk,abm->mk', G, Hhat)      # (nlb, nt)

    # Calculate laplacian per element
    edofs = basis.dofs.element_dofs               # (nlb, nt)
    u_loc = u[edofs]                              # (nlb, nt)
    lap_uh = np.sum(u_loc * laps, axis=0)         # (nt,)

    return lap_uh

def precompute_operators(basis: Basis, calculate_grad: bool = True, calculate_laplacian: bool = True):
    ref_coords   = basis.quadrature[0] 
    n_dofs_local = basis.elem.doflocs.shape[0]
    edofs        = basis.dofs.element_dofs
    list_phi      = []
    list_dphi     = []
    list_hessian  = []

    for i in range(n_dofs_local):
        out = basis.elem.lbasis(ref_coords, i)
        list_phi.append(out[0])
        list_dphi.append(out[1])
        list_hessian.append(out[2])

    phi     = np.array(list_phi)     # (n_dofs, n_quad)
    dphi    = np.array(list_dphi)    # (n_dofs, ref_dim, n_quad)
    hessian = np.array(list_hessian) # (n_dofs, ref_dim, ref_dim)
    invA    = basis.mesh.mapping().invA

    if not calculate_grad:
        grad_phi = None
    else:
        grad_phi  = np.einsum('kie, dkq -> diqe', invA, dphi) # (n_dofs, phys_dim, n_quad, n_elems)

    if not calculate_laplacian:
        laplacian_phi = None
    else:
        laplacian_phi = np.einsum('kie, lie, dkl -> de', invA, invA, hessian) # (n_dofs, n_elems)

    return edofs, phi, grad_phi, laplacian_phi

def precompute_operators_at_centroids(basis: Basis, calculate_grad: bool = True, calculate_laplacian: bool = True, dim: int = 2):
    
    if dim == 2:
        ref_coords   = np.array([1/3, 1/3])
    elif dim == 3:
        ref_coords   = np.array([1/4, 1/4, 1/4])

    n_dofs_local = basis.elem.doflocs.shape[0]
    edofs        = basis.dofs.element_dofs
    list_phi      = []
    list_dphi     = []
    list_hessian  = []

    for i in range(n_dofs_local):
        out = basis.elem.lbasis(ref_coords, i)
        list_phi.append(out[0])
        list_dphi.append(out[1])
        list_hessian.append(out[2])

    phi     = np.array(list_phi)     # (n_dofs)
    dphi    = np.array(list_dphi)    # (n_dofs, ref_dim)
    hessian = np.array(list_hessian) # (n_dofs, ref_dim, ref_dim)
    invA    = basis.mesh.mapping().invA

    if not calculate_grad:
        grad_phi = None
    else:
        grad_phi  = np.einsum('kie, dk -> die', invA, dphi) # (n_dofs, phys_dim, n_quad, n_elems)

    if not calculate_laplacian:
        laplacian_phi = None
    else:
        laplacian_phi = np.einsum('kie, lie, dkl -> de', invA, invA, hessian) # (n_dofs, n_elems)

    return edofs, phi, grad_phi, laplacian_phi