import json
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn import set_config
set_config(assume_finite=True)  # up to 10% speedup

import sys
sys.path.append('.')

from src.features.build_features import make_rff
from src.models.ridge_solvers import ridgesvd, get_beta

def single_run(run_inputs, run_params, delta_t=1, solver="voc"):
    """
    The main function for running a simulation for a fixed seed. It is wrapped by the
    run_simulation function, where inputs and parameters are handled. This function 
    contains the calculations.

    Parameters:
        run_inputs: tuple of inputs S (RFFs), and returns R already shifted once
        run_params: tuple of parameters T_list, P_dict, z_list. z_list
        delta_t: time step for retraining the model. For example, if delta_t=1, the model
            is retrained at each time step. If delta_t=10, the model is retrained every 10
            time steps, and tested in every 10th time step.

    Returns:
        beta_norm_sq: array of **squared** L2 norm of the coefficients of the model
        return_forecasts: array of return forecasts
    """
    # unpack inputs and parameters
    S, R = run_inputs
    T_list, P_dict, z_list = run_params
    min_T = min(T_list)  # usually = 12
    tmax = len(S)

    if solver != "voc":
        model_list = [[Ridge(alpha=z*T, fit_intercept=False, solver=solver) for T in T_list] for z in z_list]

    # initialize arrays for storing results. Dimensions: (ts, Ts, Ps, lambdas)
    output_shape = (tmax-min_T, len(T_list), len(P_dict["12"]), len(z_list))
    beta_norm_sq = np.full(shape=output_shape,
                           fill_value=np.nan,
                           dtype=np.float64)
    return_forecasts = np.full(shape=output_shape,
                               fill_value=np.nan,
                               dtype=np.float64)

    # helper functions for calculating standardization
    def my_std(x):
        return np.sqrt(np.sum(np.square(x - x.mean(axis=0)), axis=0)/(len(x)-1))

    def standardize(t):
        """
        Standardize training sets (one for each window T); and standardize test sets,
        which are of len=1 in this code. S is an implicit input, t is timestamp of the
        test set.
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

        # perform one standardization for all T-windows
        # standardized_sets is a list with values being pairs (trainX, forecastX)
        # and the index of the list corresponds to the index of T in T_list
        standardized_sets = standardize(t)

        for T_index, T in enumerate(T_list):
            if t <= T:
                continue

            trainX, forecastX = standardized_sets[T_index]
            trainY = R[t-T:t] # already shifted

            P_list = P_dict[str(T)] # json keys must be strings

            for P_index, P in enumerate(P_list):
                if solver == "voc":
                    if P <= T:
                        B=ridgesvd(trainY, trainX[:, :P], [z*T for z in z_list])
                    else:
                        B=get_beta(trainY, trainX[:, :P], z_list)

                    forecastY = forecastX[:P] @ B

                    # store results
                    beta_norm_sq[t-min_T, T_index, P_index, :] = np.sum(np.square(B), axis=0)
                    return_forecasts[t-min_T, T_index, P_index, :] = forecastY
                
                else:
                    for z_index, z in enumerate(z_list):
                        model = model_list[z_index][T_index]
                        model.fit(trainX[:, :P], trainY)

                        forecastY = model.predict(forecastX[:P].reshape(1, -1))
                        B = model.coef_

                        # store results
                        beta_norm_sq[t-min_T, T_index, P_index, z_index] = np.sum(np.square(B), axis=0)
                        return_forecasts[t-min_T, T_index, P_index, z_index] = forecastY[0]

    return beta_norm_sq, return_forecasts


class Simulation():
    def __init__(self, path_to_processed, path_to_outputs):
        self.path_to_processed = path_to_processed
        self.path_to_outputs = path_to_outputs

        # inputs
        data = pd.read_csv(
                self.path_to_processed+"/processed_data.csv", index_col=0, parse_dates=True)
        self.G = data.iloc[:, :-1].values # remove last column which is the target variable
        self.R = data.iloc[:, -1].values  # target variable, already shifted in prereprocessing

        # load parameters from global config file
        with open(self.path_to_outputs+"/config.json", 'r') as fp:
            self.config = json.load(fp)

        self.gamma = self.config['gamma']
        self.T_list = self.config['T_list']
        self.z_list = self.config['z_list']
        self.P_dict = self.config['P_dict']
        self.P_max = self.config['P_max']
        self.delta_t = self.config['delta_t']
        self.max_seed = self.config['max_seed']
        self.last_run_seed = self.config['last_run_seed']

        # these are the same for each run. If one wants to change them, they should stop the simulation,
        # change the config file and start the simulation again. Same for gamma and delta_t
        self.run_params = (self.T_list, self.P_dict, self.z_list)
        
        self.solver = self.config.get('solver', 'voc')
        self.config['solver'] = self.solver

    def run(self, seed):
        """
        Run a single simulation for a fixed seed. This function controls IO operations
        and calls the single_run function where the calculations are done.

        IO operations expect the following folder structure:

        path_to_processed
            └── processed_data.csv
        
        path_to_outputs
            ├── beta_norm_sq/
            ├── return_forecasts/
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

        S = make_rff(self.G, self.P_max, self.gamma, seed=seed, output_type='numpy')
        run_inputs = (S, self.R)
        
        st = time.time()

        b, r = single_run(run_inputs, self.run_params, self.delta_t, self.solver)

        et = time.time()
        elapsed_time = et - st
        print('Execution time:', elapsed_time, 'seconds')

        # write numpy arrays to files
        np.save(self.path_to_outputs +
                f"/beta_norm_sq/{seed:04}beta_norm_sq.npy", b)
        np.save(self.path_to_outputs +
                f"/return_forecasts/{seed:04}return_forecasts.npy", r)
        
        # save separate config file for each seed
        # this variable is not important for the general config but it will end up there
        self.config['seed'] = seed 
        with open(self.path_to_outputs+f"/configs/{seed:04}config.json", 'w') as fp:
            json.dump(self.config, fp)

        # update last_run_seed in the config file
        self.config['last_run_seed'] = seed

        # update general config file
        with open(self.path_to_outputs+"/config.json", 'w') as fp:
            json.dump(self.config, fp)

if __name__ == '__main__':
    # specify paths and config files only!!!
    path_to_processed = "data/processed" # specify this
    path_to_outputs = "data/interim/simulation_outputs_voc_solver" # specify this

    simulation = Simulation(path_to_processed, path_to_outputs)

    last_run_seed = simulation.last_run_seed
    max_seed = simulation.max_seed
    
    for seed in range(last_run_seed + 1, max_seed + 1):
        simulation.run(seed)
