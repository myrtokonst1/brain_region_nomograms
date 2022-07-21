import enum

import GPy as GPy
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.stats import norm

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from constants.nomogram_constants import percentiles, quantiles
from constants.processed_table_column_names import median_ages, get_brain_region_average_cn, \
    get_brain_region_percentile_cn, get_brain_region_variance_cn
from constants.ukb_table_column_names import latest_age_cn

number_of_datapoints = 1000
min_age = 0
max_age = 100
number_of_bins = (max_age - min_age)*4  # 4 points per year
median_ages_sequence = np.linspace(min_age, max_age, number_of_bins)[:, np.newaxis]


def initialize_params_dictionary(brain_region):
    gpr_params = {
        median_ages: median_ages_sequence.tolist(),
    }

    # initialize average, variance and quantile column names
    for hemisphere in brain_region.get_names():
        gpr_params[get_brain_region_average_cn(hemisphere)] = []
        gpr_params[get_brain_region_variance_cn(hemisphere)] = []

        for percentile in percentiles:
            gpr_params[get_brain_region_percentile_cn(percentile, hemisphere)] = []

    return gpr_params


def plot_gpr_model(model, y_label, title, filename, save=True, plot=True):
    model.plot()
    plt.xlabel('Age')
    plt.ylabel(y_label)
    plt.title(title)

    if save:
        plt.savefig(
            f'saved_nomograms/{filename}.png',
            dpi=600)

    if plot:
        plt.show()
    else:
        plt.close()


def perform_gaussian_process_regression(df: pd.DataFrame, brain_region: enum, sex: enum, save=False, plot=False):
    gpr_params = initialize_params_dictionary(brain_region)

    for brain_region_hemisphere in brain_region:
        # get X and y
        X = df[latest_age_cn].iloc[:number_of_datapoints]
        X = X[:, None]
        y = df[brain_region_hemisphere.get_column_name()].iloc[:number_of_datapoints]
        y = y[:, None]

        mean_brain_region_volume = X.mean()
        mean_age_difference = (pdist(y)).mean()

        kernel = GPy.kern.RBF(input_dim=1, variance=mean_brain_region_volume, lengthscale=mean_age_difference)

        model = GPy.models.GPRegression(X, y, kernel)
        display(model)

        plot_gpr_model(model,y_label=brain_region_hemisphere,
                       title=f'{brain_region_hemisphere} vs Age Pre Gaussian Process Optimization for {sex} Participants',
                       filename=f'prepocess_GPR_{brain_region_hemisphere.get_name()}_{sex.get_name()}',
                       save=save, plot=plot)

        model.optimize_parallel(messages=True)
        model.optimize_restarts(num_restarts=10)

        display(model)

        plot_gpr_model(model,y_label=brain_region_hemisphere,
                       title=f'{brain_region_hemisphere} vs Age Post Gaussian Process Optimization for {sex} Participants',
                       filename=f'post_pocess_GPR_{brain_region_hemisphere.get_name()}_{sex.get_name()}',
                       save=save, plot=plot)

        # maybe make the setting of the params a function
        mean, variance = model.predict(median_ages_sequence)
        mean = mean.flatten().tolist()
        std = [np.sqrt(v) for v in variance.flatten()]

        hemisphere_name = brain_region_hemisphere.get_name()
        gpr_params[get_brain_region_average_cn(hemisphere_name)] = mean
        gpr_params[get_brain_region_variance_cn(hemisphere_name)] = std

        for n in range(len(percentiles)):
            quantile = quantiles[n]
            percentile = percentiles[n]

            quantile_column_name = get_brain_region_percentile_cn(percentile, hemisphere_name)
            gpr_params[quantile_column_name] = norm.ppf(quantile, loc=mean, scale=std)

    return pd.DataFrame.from_dict(gpr_params)

