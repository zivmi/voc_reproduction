import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import sys 
import datetime as dt
import statsmodels.api as sm
sys.path.append('.')

from src.utils.utils import unpack_results


def plot_broken_x(df, T, z_list, y_label, save_path, ytype=None):
    f, (ax, ax2) = plt.subplots(1, 2, width_ratios=[3, 1], sharey=True)
    f.subplots_adjust(hspace=0.05)  # adjust space between axes

    df = df.loc[df['T']==T]
    x = df['c'].values
    y = df.loc[:, z_list].values

    # add vertical line at c=1
    ax.axvline(x=1,  color='0.5', linestyle='--')

    # plot the same data on both axes
    ax.plot(x, y)
    ax2.plot(x, y)

    if T==12:
        ax.set_xlim(0, 50)
        ax2.set_xlim(990, 1000)
    elif T==60:
        ax.set_xlim(0, 10)
        ax2.set_xlim(195, 200)
    elif T==120:
        ax.set_xlim(0, 10)
        ax2.set_xlim(95, 100)
    else:
        raise ValueError('T must be 12, 60 or 120')

    # in case ytype is specified, set y limits like in VoC, for comparison
    # with the original plots 
    if ytype == "r2":
        ax.set_ylim(-3, 0)
        ax2.set_ylim(-3, 0)
    elif ytype == "beta":
        ax.set_ylim(0, 3)
        ax2.set_ylim(0, 3)
    elif ytype == "er":
        ax.set_ylim(0, 0.04)
        ax2.set_ylim(0, 0.04)
    elif ytype == "vola":
        ax.set_ylim(0, 5)
        ax2.set_ylim(0, 5)
    elif ytype == "sr":
        ax.set_ylim(0, 5) # TODO to be mached ...
        ax2.set_ylim(0, 5)
    else:
        pass

    ax.set_ylabel(y_label)
    ax2.legend(z_list, title='z')
    f.suptitle(f'T={T}')

    # hide the spines between ax and ax2
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax.yaxis.tick_left()
    ax2.yaxis.tick_right()

    plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

    # set x label but place it at the center of the figure
    f.text(0.5, 0.04, 'c', ha='center', va='center')

    d = 0.02 # .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-3*d, +3*d), (1-d, 1+d), **kwargs) # 3 corrects for ratio of subplot widths
    ax2.plot((-3*d, +3*d), (-d, +d), **kwargs)

    plt.savefig(save_path, dpi=300)

    plt.close()

    return None # f, (ax, ax2)


if __name__ == '__main__':

    from_seed = 1
    to_seed = 1000
    interim_data_path = "data/interim/simulation_outputs_voc_solver"
    preprocessed_data_path = "data/processed/processed_data.csv"
    save_path = "reports/figures/curves_voc_solver"

    with open(interim_data_path + "/config.json", 'r') as fp:
        config = json.load(fp)
        last_run_seed = config['last_run_seed']
        gamma = config['gamma']
        T_list = config['T_list']
        z_list = config['z_list']
        P_dict = config['P_dict']
        P_max = 12000
        delta_t = config['delta_t']

    data = pd.read_csv(preprocessed_data_path, index_col=0, parse_dates=True)
    dates = data.index[min(T_list):]

    interim_data = pd.read_csv('data/interim/formatted_goyal_data.csv', index_col=0, parse_dates=True)
    interim_data = interim_data.loc[interim_data.index > dt.datetime(1927, 1, 1)]
    
    # reminder: interim_data.R is UNSCALED EXCESS returns of S&P500 wrt Rfree
    realized_R = interim_data.R
    sigma_R = (realized_R**2).rolling(12).mean().apply(np.sqrt) 

    run_params_unpacking = (T_list, P_dict, z_list) 
    
    first_pass = True

    for seed in range(from_seed, to_seed+1):
        print(seed)
        # load results
        b = np.load(f"{interim_data_path}/beta_norm_sq/{seed:04}beta_norm_sq.npy")
        r = np.load(f"{interim_data_path}/return_forecasts/{seed:04}return_forecasts.npy")

        # unpack results
        beta_norm_sq_df, return_forecasts_df = unpack_results((b, r), run_params_unpacking, dates)

        # filter by dates to match the original timeframe
        # TODO source this and other parameters from vis_config.json
        end_date =  dt.datetime(2025, 1, 1) 
        idx_mask = beta_norm_sq_df.index < end_date
        beta_norm_sq_df = beta_norm_sq_df.loc[idx_mask]
        return_forecasts_df = return_forecasts_df.loc[idx_mask]

        # here: position = scaled forecast of excess return 
        # alternative: position = forecasted return / beta 
        strategy_returns_rescaled = return_forecasts_df.multiply(realized_R, axis=0)

        # rescale forecasted returns because they were scaled by vola
        return_forecasts_rescaled = return_forecasts_df.multiply(sigma_R, axis=0)

        #### calculate metrics
        # beta
        mean_beta_norm_sq = beta_norm_sq_df.mean().unstack()
        beta_norm = np.sqrt(mean_beta_norm_sq) # in VoC paper they plotted square root of norm of beta
        beta_norm = beta_norm.reset_index()
        beta_norm['c'] = beta_norm['P']/beta_norm['T']

        # R2
        forecast_errors = return_forecasts_rescaled.subtract(realized_R, axis=0)
        R_sq = 1 - forecast_errors.var() / realized_R.var()
        R_sq_temp = (R_sq
                    .to_frame()
                    .unstack()
                    .droplevel(0, axis=1)
                    .reset_index()
                    )
        R_sq_temp['c'] = R_sq_temp['P']/R_sq_temp['T']
        R_sq_temp.head()

        # expected returns
        strategy_returns = strategy_returns_rescaled.mean().unstack().reset_index()
        strategy_returns['c'] = strategy_returns['P']/strategy_returns['T']
        strategy_returns.head()

        # volatility
        strategy_vola = strategy_returns_rescaled.std().unstack().reset_index()
        strategy_vola['c'] = strategy_vola['P']/strategy_vola['T']

        # sharpe ratio
        # TODO rescale to annualized sharpe ratio (annualizes expected returns over annualized volatility)
        sharpe_ratio = strategy_returns.copy()
        sharpe_ratio.loc[:, z_list] = strategy_returns.loc[:, z_list] / strategy_vola.loc[:, z_list] * np.sqrt(12)

        # alpha and alpha t-stat
        if first_pass:
            indices = strategy_returns_rescaled.columns.to_list()
            # indices are tuples (T, P, z), like (12, 100, 0.1)
        
        # initialize an empty dataframe to store alpha and alpha t-stat
        # but keep the structure the same as for all the other metrics
        alpha = strategy_vola.copy()
        alpha.loc[:,z_list] = np.zeros_like(alpha.loc[:,z_list])
        alpha_t_stat = alpha.copy()
        std_regression_residuals = alpha.copy()

        for multi_index in indices:
            Y = strategy_returns_rescaled.loc[:,multi_index].dropna()
            X = realized_R.loc[Y.index]#.dropna()
            X = sm.add_constant(X)
            model = sm.OLS(Y,X)
            results = model.fit()

            P=multi_index[1]
            T=multi_index[0]
            z=multi_index[2]

            alpha.loc[(alpha.loc[:, 'P']==P) & (alpha.loc[:, 'T']==T), z] = results.params.iloc[0]
            alpha_t_stat.loc[(alpha_t_stat.loc[:, 'P']==P) & (alpha_t_stat.loc[:, 'T']==T), z] = results.tvalues.iloc[0]
            std_regression_residuals.loc[(std_regression_residuals.loc[:, 'P']==P) & (std_regression_residuals.loc[:, 'T']==T), z] = results.resid.std()

        # information ratio
        information_ratio = alpha.copy()
        information_ratio.loc[:, z_list] = alpha.loc[:, z_list] / std_regression_residuals.loc[:, z_list]

        # std of IR
        std_information_ratio = std_regression_residuals

        # initialize the average dataframes
        # otherwise average metrics over seeds
        if first_pass:
            beta_norm_avg = beta_norm
            R_sq_avg = R_sq_temp
            strategy_returns_avg = strategy_returns
            strategy_vola_avg = strategy_vola
            sharpe_ratio_avg = sharpe_ratio
            information_ratio_avg = information_ratio
            alpha_avg = alpha
            alpha_t_stat_avg = alpha_t_stat
            std_information_ratio_avg = std_information_ratio
            first_pass = False
        else:
            # average dataframes over different seeds
            # fill_value=0 makes addition use 0 for missing values. I did not see any NaNs in the dataframes
            beta_norm_avg.loc[:,z_list] = beta_norm_avg.loc[:,z_list].add(beta_norm.loc[:,z_list], fill_value=0) 
            R_sq_avg.loc[:,z_list] = R_sq_avg.loc[:,z_list].add(R_sq_temp.loc[:,z_list], fill_value=0)
            strategy_returns_avg.loc[:,z_list] = strategy_returns_avg.loc[:,z_list].add(strategy_returns.loc[:,z_list], fill_value=0)
            strategy_vola_avg.loc[:,z_list] = strategy_vola_avg.loc[:,z_list].add(strategy_vola.loc[:,z_list], fill_value=0)
            sharpe_ratio_avg.loc[:,z_list] = sharpe_ratio_avg.loc[:,z_list].add(sharpe_ratio.loc[:,z_list], fill_value=0)
            information_ratio_avg.loc[:,z_list] = information_ratio_avg.loc[:,z_list].add(information_ratio.loc[:,z_list], fill_value=0)
            alpha_avg.loc[:,z_list] = alpha_avg.loc[:,z_list].add(alpha.loc[:,z_list], fill_value=0)
            alpha_t_stat_avg.loc[:,z_list] = alpha_t_stat_avg.loc[:,z_list].add(alpha_t_stat.loc[:,z_list], fill_value=0)
            std_information_ratio_avg.loc[:,z_list] = std_information_ratio_avg.loc[:,z_list].add(std_information_ratio.loc[:,z_list], fill_value=0)

    beta_norm_avg.loc[:,z_list] = beta_norm_avg.loc[:,z_list] / (to_seed)
    R_sq_avg.loc[:,z_list] = R_sq_avg.loc[:,z_list] / (to_seed )
    strategy_returns_avg.loc[:,z_list] = strategy_returns_avg.loc[:,z_list] / (to_seed )
    strategy_vola_avg.loc[:,z_list] = strategy_vola_avg.loc[:,z_list] / (to_seed )
    sharpe_ratio_avg.loc[:,z_list] = sharpe_ratio_avg.loc[:,z_list] / (to_seed )
    information_ratio_avg.loc[:,z_list] = information_ratio_avg.loc[:,z_list] / (to_seed )
    alpha_avg.loc[:,z_list] = alpha_avg.loc[:,z_list] / (to_seed )
    alpha_t_stat_avg.loc[:,z_list] = alpha_t_stat_avg.loc[:,z_list] / (to_seed )
    std_information_ratio_avg.loc[:,z_list] = std_information_ratio_avg.loc[:,z_list] / (to_seed )

    # plot averaged dataframes
    for T in T_list:
        plot_broken_x(beta_norm_avg, T, z_list, 
                    y_label=r'$\| \hat {\beta}\|$', 
                    save_path=f"{save_path}/beta-t{T}.png")
        plot_broken_x(R_sq_avg, T, z_list,
                        r'$R_{OSS}^2$',
                        f"{save_path}/r2-t{T}.png")
        plot_broken_x(strategy_returns_avg, T, z_list,
                        "Expected Return",
                        f"{save_path}/er-t{T}.png")
        plot_broken_x(strategy_vola_avg, T, z_list,
                        "Volatility",
                        f"{save_path}/vola-t{T}.png")
        plot_broken_x(sharpe_ratio_avg, T, z_list,
                        "Sharpe Ratio",
                        f"{save_path}/sr-t{T}.png")
        plot_broken_x(information_ratio_avg, T, z_list,
                        "Information Ratio",
                        f"{save_path}/ir-t{T}.png")
        plot_broken_x(alpha_avg, T, z_list,
                        r'$\alpha$',
                        f"{save_path}/alpha-t{T}.png")
        plot_broken_x(alpha_t_stat_avg, T, z_list,
                        r'$\alpha$ t-statistic',
                        f"{save_path}/alpha-t-stat-t{T}.png")
        plot_broken_x(std_information_ratio_avg, T, z_list,
                        "stddev(IR)",
                        f"{save_path}/std-ir-t{T}.png")