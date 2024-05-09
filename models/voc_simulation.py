import json
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn import set_config
set_config(assume_finite=True)  # up to 10% speedup


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
    omegas = np.random.normal(0, 1, (15, int(P/2)))
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


def single_run(run_inputs, run_params, delta_t=1):
    """
    The main function for running a simulation for a fixed seed. It is wrapped by the
    run_simulation function, where inputs and parameters are handled. This function 
    contains the calculations.

    Parameters:
        run_inputs: tuple of inputs S (RFFs), and returns R already shifted once
        run_params: tuple of parameters T_list, P_dict, model_dict. model_dict is 
            a dictionary of models for each T in T_list. Each model is a list of Ridge
            regressions with different lambdas. Lambda is determined by T*z, where z is
            a parameter that is tracked by the index of the model in the list. P_dict contains
            lists of number of features to use for each T. 
        delta_t: time step for retraining the model. For example, if delta_t=1, the model
            is retrained at each time step. If delta_t=10, the model is retrained every 10
            time steps.

    Returns:
        beta_norm_sq: array of **squared** L2 norm of the coefficients of the model
        return_forecasts: array of return forecasts
        strategy_returns: array of strategy returns. Strategy returns are the product of
            return forecasts and actual returns at each time step, or in other words the
            trading strategy takes position in the index equal to the return forecast.
    """
    # unpack inputs and parameters
    S, R = run_inputs
    T_list, P_dict, model_dict = run_params
    min_T = min(T_list)  # usually =12
    num_of_models = len(model_dict[T_list[0]])
    num_of_Ps = len(P_dict["12"])
    tmax = len(S)

    # initialize arrays for storing results. Dimensions: (ts, Ts, Ps, lambdas)
    output_shape = (tmax-min_T, len(T_list), num_of_Ps, num_of_models)
    beta_norm_sq = np.full(shape=output_shape,
                           fill_value=np.nan,
                           dtype=np.float64)
    return_forecasts = np.full(shape=output_shape,
                               fill_value=np.nan,
                               dtype=np.float64)
    strategy_returns = np.full(shape=output_shape,
                               fill_value=np.nan,
                               dtype=np.float64)

    # helper functions for calculating standardization
    def my_std(x):
        return np.sqrt(np.sum(np.square(x - x.mean(axis=0)), axis=0)/(len(x)-1))

    def standardize(t):
        """
        Standardize training sets (one for each window T) and test set of len=1,
        at time t. It takes S as implicit input.
        """
        sets = []

        for T in T_list:
            if t <= T:
                sets.append((np.nan, np.nan))
            else:
                training_std = my_std(S[t-T:t])
                trainX = S[t-T:t] / training_std
                forecastX = S[t] / training_std
                sets.append((trainX, forecastX))

        return sets

    for t in range(min_T, tmax, delta_t):
        # print progress
        if t % 100 == 0:
            print(f"progress: {t/(tmax-min_T):2.1%}")

        # perform one standardization for all Ts
        standardized_sets = standardize(t)

        for T_index, T in enumerate(T_list):
            if t <= T:
                continue

            trainX, forecastX = standardized_sets[T_index]
            trainY = R[t-T:t] # already shifted

            P_list = P_dict[str(T)] # json keys must be strings
            model_list = model_dict[T]
            grid = [(P_index, model_index)
                for P_index in range(num_of_Ps)
                for model_index in range(num_of_models)]

            for P_index, model_index in grid:
                P = P_list[P_index]
                # get model of appropriate shrinkage lambda=T*z, z is tracked by model_index
                # the value of z is equal to z_list[model_index]
                model = model_list[model_index]

                # train models using the first P features from S matrix
                model.fit(trainX[:, :P], trainY)
                forecastY = model.predict(forecastX[:P].reshape(1, -1))

                # store results
                beta_norm_sq[t-min_T, T_index, P_index,
                                model_index] = np.sum(np.square(model.coef_))
                return_forecasts[t-min_T, T_index,
                                    P_index, model_index] = forecastY[0]
                strategy_returns[t-min_T, T_index, P_index,
                                    model_index] = forecastY[0] * R[t]

    return beta_norm_sq, return_forecasts, strategy_returns


def run_simulation(seed, path_to_processed, path_to_outputs):
    """
    Run a single simulation for a fixed seed. This function controls IO operations
    and calls the single_run function where the calculations are done.

    The function requires the following folder structure:

    path_to_processed
        └── processed_data.csv
    
    path_to_outputs
        ├── beta_norm_sq/
        ├── return_forecasts/
        ├── strategy_returns/
        ├── configs/
        └── config.json

    Parameters:
        seed: random seed
        path_to_processed: path to the folder with the preprocessed data
        path_to_outputs: path to the folder interim outputs and config.json are stored

    Returns:
        None. The function writes the results to the files in the specified folders.
    """
    print(f"seed: {seed}")
    data = pd.read_csv(
            path_to_processed+"/processed_data.csv", index_col=0, parse_dates=True)

    # load parameters from config file
    with open(path_to_outputs+"/config.json", 'r') as fp:
        config = json.load(fp)
        gamma = config['gamma']
        T_list = config['T_list']
        z_list = config['z_list']
        P_dict = config['P_dict']
        P_max = config['P_max']
        delta_t = config['delta_t']
        solver = config['solver']
        fit_intercept = config['fit_intercept']

    # create dictionary of models for each T in T_list
    model_dict = {T: [Ridge(alpha=T*z, fit_intercept=fit_intercept, solver=solver) 
                        for z in z_list] for T in T_list}

    # inputs
    G = data.iloc[:, :-1].values # remove last column which is the target variable
    R = data.iloc[:, -1].values  # target variable, already shifted in prereprocessing
    S = make_rff(G, P_max, gamma=gamma, seed=seed, output_type='numpy')
    run_inputs = (S, R)
    run_params = (T_list, P_dict, model_dict)

    st = time.time()

    b, r, sr = single_run(run_inputs, run_params, delta_t)

    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

    # write numpy arrays to files
    np.save(path_to_outputs +
            f"/beta_norm_sq/{seed:04}beta_norm_sq.npy", b)
    np.save(path_to_outputs +
            f"/return_forecasts/{seed:04}return_forecasts.npy", r)
    np.save(path_to_outputs+
            f"/strategy_returns/{seed:04}strategy_returns.npy", sr)
    
    # save separate config file for each seed
    config['seed'] = seed
    with open(path_to_outputs+f"/configs/{seed:04}config.json", 'w') as fp:
        json.dump(config, fp)

    # update last_run_seed in the config file
    config['last_run_seed'] = seed

    # update general config file
    with open(path_to_outputs+"/config.json", 'w') as fp:
        json.dump(config, fp)

    return None


if __name__ == '__main__':
    path_to_processed = "data/processed" # specify this
    path_to_outputs = "data/interim/final2_simulation_outputs" # specify this

    # load the last and the max seeds from the config file
    with open(path_to_outputs+"/config.json", 'r') as fp:
        config = json.load(fp)
        last_run_seed = config['last_run_seed']
        max_seed = config['max_seed']
    
    for seed in range(last_run_seed + 1, max_seed + 1):
        run_simulation(seed, path_to_processed, path_to_outputs)
