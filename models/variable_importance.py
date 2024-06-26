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
        vars_to_exclude = config['vars_to_exclude']


    # for each variable in exclude list, remove it from the data
    # and perform all the procedures as in the original procedure

    for variable in vars_to_exclude:
        var_id = data.columns.get_loc(variable)
        data_temp = data.drop(variable, axis=1)

        # inputs
        G = data_temp.iloc[:, :-1].values # remove last column which is the target variable
        R = data_temp.iloc[:, -1].values  # target variable, already shifted in prereprocessing
        S = make_rff(G, P_max, gamma=gamma, seed=seed, output_type='numpy')
        run_inputs = (S, R)
        run_params = (T_list, P_dict, z_list)

        st = time.time()

        b, r = single_run(run_inputs, run_params, delta_t)

        et = time.time()
        elapsed_time = et - st
        print('Execution time:', elapsed_time, 'seconds')

        # write numpy arrays to files
        np.save(path_to_outputs +
                f"/beta_norm_sq/{seed:04}_var{var_id:02}_beta_norm_sq.npy", b)
        np.save(path_to_outputs +
                f"/return_forecasts/{seed:04}_var{var_id:02}_return_forecasts.npy", r)

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
    global_time = time.time()
    path_to_processed = "data/processed" # specify this
    path_to_outputs = "data/interim/variable_importance" # specify this

    # load the last and the max seeds from the config file
    with open(path_to_outputs+"/config.json", 'r') as fp:
        config = json.load(fp)
        last_run_seed = config['last_run_seed']
        max_seed = config['max_seed']
    
    for seed in range(last_run_seed + 1, max_seed + 1):
        run_simulation(seed, path_to_processed, path_to_outputs)
    print('Total execution time:', time.time() - global_time, 'seconds')