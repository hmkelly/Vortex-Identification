"""
Created on Tue Aug 06 11:42:00 2024

@author: Harley Kelly
@email: h.kelly21@imperial.ac.uk
"""

import numpy as np

"This function solves the eigenvalue problem for the lambda criteria"
def lambda_criterion(lambda_matrix):
    # Flatten the array and transpose it to prepare for eigvals function
    sp = lambda_matrix.shape
    lambda_matrix_flat = lambda_matrix.reshape(sp[0], sp[1], sp[2] * sp[3] * sp[4])
    lambda_matrix_flat = lambda_matrix_flat.transpose(2, 0, 1)

    # Find the eigenvalues and then only keep the lambda2 value
    eigs_mat_flat = np.linalg.eigvals(lambda_matrix_flat).transpose(1, 0)
    eigs_mat = eigs_mat_flat.reshape(3, sp[2], sp[3], sp[4])

    return np.sort(eigs_mat, axis=0)[1, :, :, :]

"""
Calculate the incompressible Q criterion for a vortex using the velocity vector field
of shape (v,x,y,z) where v is the number of components of the velocity vector field.
This method will take about 40 seconds to run on a 480x320x320 grid.
"""

def calc_Q_criterion(V_arr, x=1., y=1., z=1.):
    # Define the gradient tensor
    V_grad = np.gradient(V_arr,  x, y, z, axis=(1,2,3)) # units of s-1

    # Define the Velocity Strain rate tensor
    S_V = 0.5 * ( V_grad + np.transpose(V_grad, (1,0,2,3,4)) ) # units of s-1

    # Define the Velocity Rotation rate tensor
    Omega_V = 0.5 * ( V_grad - np.transpose(V_grad, (1,0,2,3,4)) ) # units of s-1

    # Q = 1/2 ( ||Omega|| ^ 2 - ||S||^2 )
    Q_incomp = 0.5 * ( np.square( np.linalg.norm(Omega_V, axis=(0,1), ord='fro') ) - np.square( np.linalg.norm(S_V, axis=(0,1), ord='fro') ) )

    return Q_incomp

"""
Calculate the lambda2 criterion for a vortex
"""

def calc_lambda2(V_arr, x=1., y=1., z=1.):
    # Define the gradient tensor
    V_grad = np.gradient(V_arr,  x, y, z, axis=(1,2,3)) # units of s-1

    # Define the Velocity Strain rate tensor
    S_V = 0.5 * ( V_grad + np.transpose(V_grad, (1,0,2,3,4)) ) # units of s-1

    # Define the Velocity Rotation rate tensor
    Omega_V = 0.5 * ( V_grad - np.transpose(V_grad, (1,0,2,3,4)) ) # units of s-1

    # Calculate the S^2 and Omega^2 tensors
    S2_O2 = np.einsum('ik...,kj...->ij...', S_V, S_V) + np.einsum('ik...,kj...->ij...', Omega_V, Omega_V) # units of s-2

    return lambda_criterion(S2_O2)

"""
Calculate the weighted lambda2 criterion for a vortex
"""

def calc_weighted_lambda2(V_arr, Rho_arr, x=1., y=1., z=1.):
    # Define the gradient tensor
    V_grad = np.gradient(V_arr,  x, y, z, axis=(1,2,3)) # units of s-1

    # Define the Velocity Strain rate tensor
    S_V = 0.5 * ( V_grad + np.transpose(V_grad, (1,0,2,3,4)) ) # units of s-1

    # Define the Velocity Rotation rate tensor
    Omega_V = 0.5 * ( V_grad - np.transpose(V_grad, (1,0,2,3,4)) ) # units of s-1

    # Calculate the S^2 and Omega^2 tensors
    S2_O2 = np.einsum('ik...,kj...->ij...', S_V, S_V) + np.einsum('ik...,kj...->ij...', Omega_V, Omega_V) # units of s-2

    # Weight the S2_O2 tensor by the density field
    WL2 = np.einsum('xyz,ijxyz->ijxyz', Rho_arr, S2_O2)

    return lambda_criterion(WL2) 

"""
Calculate the lambda rho criterion for a vortex
"""

def calc_lambda_rho(V_arr, Rho_arr, x=1., y=1., z=1.):
    # Define the gradient tensor
    V_grad = np.gradient(V_arr,  x, y, z, axis=(1,2,3)) # units of s-1

    # Define the Velocity Strain rate tensor
    S_V = 0.5 * ( V_grad + np.transpose(V_grad, (1,0,2,3,4)) ) # units of s-1

    # Define the Velocity Rotation rate tensor
    Omega_V = 0.5 * ( V_grad - np.transpose(V_grad, (1,0,2,3,4)) ) # units of s-1

    # Calculate the S^2 and Omega^2 tensors
    S2_O2 = np.einsum('ik...,kj...->ij...', S_V, S_V) + np.einsum('ik...,kj...->ij...', Omega_V, Omega_V) # units of s-2

    # Weight the S2_O2 tensor by the density field
    WL2 = np.einsum('xyz,ijxyz->ijxyz', Rho_arr, S2_O2)

    # Calculate Term 1
    grad_Rho = np.gradient(Rho_arr, x, y, z, axis=(0,1,2)) # units of kg m-4

    A = np.einsum('i...,jk...->ij...', V_arr, V_grad) # units of m s-2

    Inhom = 0.5 * np.einsum('k...,ij...->ij...', grad_Rho , (A + np.transpose(A, (1,0,2,3,4))))

    term_1 = WL2 + Inhom # units of kg m-3 s-2

    # Calculate Term 2
    theta = np.einsum('kkxyz->xyz', V_grad) # units of s-1

    A = np.einsum('xyz,ixyz->ixyz', Rho_arr, V_arr) # units of kg m-2 s-1

    B = np.einsum('xyz,ixyz->ixyz', theta, A) # units of kg m-2 s-2

    C = np.gradient(B, x, y, z, axis=(1,2,3)) # units of kg m-3 s-2

    term_2 = 0.5 * (C + np.transpose(C, (1,0,2,3,4))) # units of kg m-3 s-2

    return lambda_criterion(term_1 + term_2)