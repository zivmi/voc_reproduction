import numpy as np
import pandas as pd

def unpack_results(results, run_params, date_index):
    beta_norm_sq, return_forecasts = results
    T_list, P_dict, z_list = run_params

    Plen = len(P_dict[str(T_list[0])])

    if type(z_list) == dict:
        raise ValueError('z_list must be a list, not a dictionary. Use `run_params_unpacking`, not `run_params`')

    # make dummy dataframes to be populated
    list_beta_norm_sq = {}
    list_return_forecasts = {}

    # for T in T_list make dataframes
    for T_index, T in enumerate(T_list):
        list_beta_norm_sq.update({T: pd.DataFrame(data=beta_norm_sq[:,T_index, :, :].reshape(-1, Plen*len(z_list)), 
                                        index=date_index, 
                                        columns=pd.MultiIndex.from_product([P_dict[str(T)], z_list], 
                                                                        names=['P', 'z']))}) 
        list_return_forecasts.update({T: pd.DataFrame(data=return_forecasts[:,T_index, :, :].reshape(-1, Plen*len(z_list)), 
                                            index=date_index, 
                                            columns=pd.MultiIndex.from_product([P_dict[str(T)], z_list], 
                                                                            names=['P', 'z']))})

    beta_norm_sq_df = pd.concat(list_beta_norm_sq, axis=1)
    return_forecasts_df = pd.concat(list_return_forecasts, axis=1)
    
    beta_norm_sq_df.columns.names = ['T', 'P', 'z']
    return_forecasts_df.columns.names = ['T', 'P', 'z']
    
    return beta_norm_sq_df, return_forecasts_df