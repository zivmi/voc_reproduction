import numpy as np
from scipy.linalg import svd

def ridgesvd(Y, X, lambda_list):
    """
    Computes ridge regression coefficients using singular value decomposition.
    
    Parameters:
        Y : array_like
            Target vector of shape (T,).
        X : array_like
            Design matrix of shape (T, P).
        lambda_list : array_like
            Array of ridge regularization parameters of shape (L,).
    
    Returns:
        B : ndarray
            Ridge regression coefficients of shape (P, L).
    """
    if np.isnan(X).sum() + np.isnan(Y).sum() > 0:
        raise ValueError("Missing data")

    L = len(lambda_list)
    # MATLAB uses 'gesvd', default is 'gesdd'
    U, d, Vt = svd(X, check_finite=False, lapack_driver='gesvd') 
    T, P = X.shape

    if T >= P:
        compl = np.zeros((P, T - P))
    else:
        compl = np.zeros((P - T, T))
    
    B = np.zeros((P, L))

    for l, lam in enumerate(lambda_list):
        if T >= P:
            B[:, l] = Vt.T @ np.hstack((np.diag(d / (d**2 + lam)), compl)) @ U.T @ Y
        else:
            B[:, l] = Vt.T @ np.vstack((np.diag(d / (d**2 + lam)), compl)) @ U.T @ Y
    
    return B

def get_beta(Y, X, z_list):
    """
    Computes beta coefficients using ridge regression with SVD.
    
    Parameters:
        Y : array_like
            Target vector of shape (T,).
        X : array_like
            Features matrix of shape (T, P).
        z_list : array_like
            Array of ridge regularization parameters of shape (L,).
    
    Returns:
        B : ndarray
            Ridge regression coefficients of shape (P, L).
    """
    if np.isnan(X).sum() + np.isnan(Y).sum() > 0:
        raise ValueError("Missing data")
    
    L_ = len(z_list)
    T_ = X.shape[0]
    P_ = X.shape[1]

    if P_ > T_:
        a_matrix = X @ X.T / T_  # T_ x T_
    else:
        a_matrix = X.T @ X / T_  # P_ x P_

    U_a, d_a, _ = svd(a_matrix, check_finite=False, lapack_driver='gesvd')
    scale_eigval = ((d_a * T_)**(-1/2))

    # originally only the X.T version was implemented, but that
    # causes error in multiplication of matrices even in the original
    # MATLAB code. The second brach (P<=T) is not used because in that
    # case we call 'ridgesvd' instead of 'get_beta'.
    if P_ > T_:
        W = X.T @ U_a @ np.diag(scale_eigval)
    else:
        W = X @ U_a @ np.diag(scale_eigval)

    a_matrix_eigval = d_a #.reshape(-1, 1)  # P_ x 1
    
    # FIXME the following code does not run for P<=T
    # Fix it so that it runs. 
    signal_times_return = X.T @ Y / T_  # (SR): M x 1
    signal_times_return_times_v = W.T @ signal_times_return  # V' * (SR): T_ x 1

    B = np.zeros((P_, L_))
    for l, lam in enumerate(z_list):
        B[:, l] = W @ np.diag(1 / (a_matrix_eigval + lam)) @ signal_times_return_times_v

    return B