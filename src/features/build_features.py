import numpy as np
import pandas as pd


def make_rff(G, P, gamma=2, seed=59148, output_type='numpy'):
    """
    Generate P Random Fourier Features from the set of independent variables G.

    Parameters:
        G: 2d numpy array of independent variables, shape (n_samples, n_features)
        where n_samples is the number of timestamps 
        P: number of RFFs to generate
        gamma: scaling factor in the sin and cos functions 
        seed: random seed 
        output_type: 'numpy' or 'pandas'

    Returns:
        S: matrix of random features, the shape of S is (len(G), P)
    """
    np.random.seed(seed)
    num_of_independent_vars = G.shape[1]
    omegas = np.random.normal(0, 1, (num_of_independent_vars, int(P/2)))
    A = gamma * G @ omegas
    S_sin = np.sin(A)
    S_cos = np.cos(A)
    S = np.full((S_sin.shape[0], S_sin.shape[1] +
                S_cos.shape[1]), fill_value=np.nan, dtype=S_sin.dtype)
    S[:, 0::2] = S_sin
    S[:, 1::2] = S_cos

    if output_type == 'pandas':
        S = pd.DataFrame(data=S, index=G.index, columns=np.arange(P))
    elif output_type == 'numpy':
        return S
    else:
        raise ValueError('output_type must be "pandas" or "numpy"')
