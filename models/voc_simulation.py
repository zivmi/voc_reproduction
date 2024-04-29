#!/home/miroslav/miniforge3/envs/voc python

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import os
from sklearn import set_config
set_config(assume_finite=True) # up to 10% speedup
import time
import json

def make_rff(G, P, gamma=2, seed=59148, output_type='numpy'):
    np.random.seed(seed)
    omegas = np.random.normal(0, 1, (15, int(P/2)))
    A = gamma * G @ omegas
    S_sin = np.sin(A) 
    S_cos = np.cos(A)
    S = np.full((S_sin.shape[0], S_sin.shape[1] + S_cos.shape[1]), fill_value=np.nan, dtype=S_sin.dtype)
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
    Run the backtest for a single run.
    S: matrix of features
    R: vector of target variable, shifted once 
    T_list: list of training window lengths
    model_list: list of models to train (ridge regressions with different lambdas)
    P_list: list of number of features to use for each model
    delta_t: time step for retraining the model

    Returns:
    """

    S, R = run_inputs
    T_list, P_list, model_dict = run_params

    # initialize arrays for storing results
    num_of_models = len(model_dict[T_list[0]])

    min_T = min(T_list) # usually =12

    # dimensions: (ts, Ts, Ps, lambdas)
    output_shape = (len(S)-min_T, len(T_list), len(P_list), num_of_models)
    beta_norm_sq = np.full(shape=output_shape, 
                    fill_value=np.nan, 
                    dtype=np.float64)
    return_forecasts = np.full(shape=output_shape, 
                               fill_value=np.nan, 
                               dtype=np.float64)
    strategy_returns = np.full(shape=output_shape, 
                               fill_value=np.nan, 
                               dtype=np.float64)

    # initialize arrays for storing intermediate variables
    # training_std = np.full((len(T_arr), S.shape[1]), fill_value=np.nan, dtype=np.float64)

    def my_std(x):
        return np.sqrt(np.sum(np.square(x - x.mean(axis=0)))/len(x))

    def standardize(t):
        training_sets = []

        for T in T_list:

            # this take a lot of time to compute
            # train_std = trainX.std(axis=0) 
            # train_std = np.sqrt(np.sum((trainX - train_mean)**2, axis=0)/T) # this is equivalent to trainX.std(axis=0)
            if t-T < 1:
                training_sets.append((np.nan, np.nan))
            else:
                training_std = my_std(S[t-T:t])
                trainX = S[t-T:t] / training_std
                forecastX = S[t] / training_std
                training_sets.append((trainX, forecastX))

        return training_sets

    grid = [(P_index, model_index) 
            for P_index in range(len(P_list)) 
            for model_index in range(num_of_models)]
    

    for t in range(min_T, len(S), delta_t):
        # print progress
        if t%100==0:
            print(f"progress: {t/(len(S)-min_T):2.1%}")

        # one standardization for all models, all complexities
        training_sets = standardize(t)

        for P_index, model_index in grid:
            for T_index, T in enumerate(T_list):
                if t-T < 1:
                    continue
                else: 
                    P = P_list[P_index]
                    # get model of appropriate shrinkage lambda=T*z, z is tracked by model_index
                    model = model_dict[T][model_index]
                    
                    trainX, forecastX = training_sets[T_index]
                    trainY = R[t-T:t]
                    # take first P features for training
                    model.fit(trainX[:,:P], trainY) 
                    forecastY = model.predict(forecastX[:P].reshape(1,-1))

                    # store results 
                    beta_norm_sq[t-min_T, T_index, P_index, model_index] = np.sum(np.square(model.coef_))
                    return_forecasts[t-min_T, T_index, P_index, model_index] = forecastY[0]
                    strategy_returns[t-min_T, T_index, P_index, model_index] = forecastY[0] * R[t]

    return beta_norm_sq, return_forecasts, strategy_returns

def run_simulation(seed):
    print(f"seed: {seed}")
    S = make_rff(G, P_max, gamma=gamma, seed=seed, output_type='numpy')
    run_inputs = (S, R)
    run_params = (T_list, P_list, model_dict)

    st = time.time()

    b, r, sr = single_run(run_inputs, run_params, delta_t)

    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

    # write numpy arrays to files
    np.save(path_to_data+f"/interim/simulation_outputs/beta_norm_sq/{seed:04}beta_norm_sq.npy", b)
    np.save(path_to_data+f"/interim/simulation_outputs/return_forecasts/{seed:04}return_forecasts.npy", r)
    np.save(path_to_data+f"/interim/simulation_outputs/strategy_returns/{seed:04}strategy_returns.npy", sr)

    # write configuration to file

    #config['last_run_seed'] = seed
    config['time_elapsed'] = elapsed_time

    with open(path_to_data+"/interim/simulation_outputs/config.json", 'w') as fp:
        json.dump(config, fp)

    return None


if __name__ == '__main__':
    path_to_data = "data"
    data = pd.read_csv(path_to_data+"/processed/processed_data.csv", index_col=0, parse_dates=True)

    # load parameters from config file
    with open(path_to_data+"/interim/simulation_outputs/config.json", 'r') as fp:
        config = json.load(fp)
        last_run_seed = config['last_run_seed']
        gamma = config['gamma']
        T_list = config['T_list']
        z_list = config['z_list']
        P_list = config['P_list']
        P_max = max(P_list)
        delta_t = config['delta_t']
        max_seed = config['max_seed']
        model_dict = {T: [Ridge(alpha=np.square(T*z), fit_intercept=True) for z in z_list] for T in T_list}
        # print the solver used in the ridge regression
        print(f"Solver used in ridge regression: {model_dict[T_list[0]][0].solver}")

    # inputs
    G = data.iloc[:,:-1].values # remove last column which is the target variable
    R = data.iloc[:,-1].values # target variable, it is already shifted

    paralellize = False

    # number of threads to used
    num_threads = 3

    if paralellize:
        from multiprocessing import Pool

        for seed in range(last_run_seed + 1, max_seed + 1, num_threads):
            with Pool(num_threads) as p:
                # take num_threads seeds from last_run_seed+1 to last_run_seed+num_threads and run simulations in parallel
                p.map(run_simulation, [seed + i for i in range(num_threads)])

                # check if all simulations are done and updated the last_run_seed in the config file
                # Check if all simulations are done
                completed_simulations = range(seed, seed + num_threads)
                all_simulations_done = all(os.path.exists(f"{path_to_data}/interim/simulation_outputs/strategy_returns/{seed:04}strategy_returns.npy") for seed in completed_simulations)

                # Update the last_run_seed in the config file
                if all_simulations_done:
                    config['last_run_seed'] = last_run_seed + num_threads
                    with open(path_to_data+"/interim/simulation_outputs/config.json", 'w') as fp:
                        json.dump(config, fp)
                else: 
                    print('Not all simulations are done')
                    print('Seeds not done:', [seed for seed in completed_simulations if not os.path.exists(f"{path_to_data}/interim/simulation_outputs/strategy_returns/{seed:04}strategy_returns.npy")])

                    # if not all simulations are done, update the last_run_seed in the config file to the minimal seed that is not done
                    config['last_run_seed'] = min(seed for seed in completed_simulations if not os.path.exists(f"{path_to_data}/interim/simulation_outputs/strategy_returns/{seed:04}strategy_returns.npy"))
                    with open(path_to_data+"/interim/simulation_outputs/config.json", 'w') as fp:
                        json.dump(config, fp)

                print('All simulations done for seeds:', all_simulations_done)
    
    else:
        for seed in range(last_run_seed + 1, max_seed + 1):
            run_simulation(seed)
            config['last_run_seed'] = seed            
            with open(path_to_data+"/interim/simulation_outputs/config.json", 'w') as fp:
                json.dump(config, fp)