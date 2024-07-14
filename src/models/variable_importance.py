import json
import time
import numpy as np
import pandas as pd
from scipy.linalg import svd
from sklearn.linear_model import Ridge
from sklearn import set_config
set_config(assume_finite=True)  # up to 10% speedup

import sys
sys.path.append('.')

from src.features.build_features import make_rff
from voc_simulation import single_run

class Simulation():
    def __init__(self, path_to_processed, path_to_outputs):
        self.path_to_processed = path_to_processed
        self.path_to_outputs = path_to_outputs

        # inputs
        self.data = pd.read_csv(
                self.path_to_processed+"/processed_data.csv", index_col=0, parse_dates=True)

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

        self.vars_to_exclude = self.config['vars_to_exclude']

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


        for variable in self.vars_to_exclude:
            var_id = self.data.columns.get_loc(variable)
            data_temp = self.data.drop(variable, axis=1)

            # inputs
            G = data_temp.iloc[:, :-1].values # remove last column which is the target variable
            R = data_temp.iloc[:, -1].values  # target variable, already shifted in prereprocessing
            S = make_rff(G, self.P_max, self.gamma, seed=seed, output_type='numpy')
            run_inputs = (S, R)

            st = time.time()

            b, r = single_run(run_inputs, self.run_params, self.delta_t)

            et = time.time()
            elapsed_time = et - st
            print('Execution time:', elapsed_time, 'seconds')

            # write numpy arrays to files
            np.save(self.path_to_outputs +
                    f"/beta_norm_sq/{seed:04}_var{var_id:02}_beta_norm_sq.npy", b)
            np.save(self.path_to_outputs +
                    f"/return_forecasts/{seed:04}_var{var_id:02}_return_forecasts.npy", r)

        # save separate config file for each seed
        self.config['seed'] = seed
        with open(self.path_to_outputs+f"/configs/{seed:04}config.json", 'w') as fp:
            json.dump(self.config, fp)

        # update last_run_seed in the config file
        self.config['last_run_seed'] = seed

        # update general config file
        with open(self.path_to_outputs+"/config.json", 'w') as fp:
            json.dump(self.config, fp)
            

if __name__ == '__main__':
    global_time = time.time()
    path_to_processed = "data/processed" # specify this
    path_to_outputs = "data/interim/variable_importance" # specify this

    simulation = Simulation(path_to_processed, path_to_outputs)

    last_run_seed = simulation.last_run_seed
    max_seed = simulation.max_seed
    
    for seed in range(last_run_seed + 1, max_seed + 1):
        simulation.run(seed)
    print('Total execution time:', time.time() - global_time, 'seconds')