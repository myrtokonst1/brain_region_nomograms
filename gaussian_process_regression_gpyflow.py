import enum

import gpflow
import tensorflow as tf
import matplotlib.pyplot as plt
from gpflow.ci_utils import ci_niter
from gpflow.utilities import print_summary, to_default_float
from scipy.stats import norm

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from tensorflow.python.client import device_lib

from constants.nomogram_constants import percentiles, quantiles
from constants.processed_table_column_names import median_ages, get_brain_region_average_cn, \
    get_brain_region_percentile_cn, get_brain_region_variance_cn
from constants.ukb_table_column_names import latest_age_cn
from enums.brain_regions.hippocampal_volume import HippocampalVolume

number_of_datapoints = 2000
min_age = 35
max_age = 95
number_of_bins = (max_age - min_age)*4  # 4 points per year
median_ages_sequence = np.linspace(min_age, max_age, number_of_bins)


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


def plot_gpr_model(X, Y, xx, mean, var, y_label, title, filename, save=True, plot=True):
    plt.plot(X, Y, "kx", mew=2)
    plt.plot(xx, mean, "C0", lw=2)
    plt.fill_between(
        xx[:, 0],
        mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
        mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
        color="C0",
        alpha=0.2,
    )

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
    brain_region_hemisphere = HippocampalVolume.LEFT
    # for brain_region_hemisphere in brain_region:
    # get X and y
    X = df[latest_age_cn].iloc[:number_of_datapoints].to_numpy()
    X = X.reshape(-1,1)
    y = df[brain_region_hemisphere.get_column_name()].iloc[:number_of_datapoints].to_numpy()
    y = y.reshape(-1,1)

    mean_brain_region_volume = np.mean(X)
    mean_age_difference = np.mean(pdist(y))

    kernel = gpflow.kernels.RBF(variance=mean_brain_region_volume, lengthscales=mean_age_difference)

    model = gpflow.models.GPR(data=(X, y), kernel=kernel)
    print_summary(model)

    for _ in range(ci_niter(10)):
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=1000))

    print_summary(model)

    mean, variance = model.predict_y(median_ages_sequence[:, np.newaxis])
    mean = mean.numpy().flatten()
    std = np.sqrt(variance.numpy()).flatten()

    hemisphere_name = brain_region_hemisphere.get_name()
    gpr_params[get_brain_region_average_cn(hemisphere_name)] = mean.tolist()
    gpr_params[get_brain_region_variance_cn(hemisphere_name)] = std.tolist()

    plt.plot(X, y, "x", mew=2, color='#E1E4EA')
    plt.plot(median_ages_sequence[:, np.newaxis], mean, "C0", lw=2)
    plt.fill_between(
        median_ages_sequence[:, np.newaxis][:, 0],
        mean - 1.96*std,
        mean + 1.96*std,
        color="#54E6FC",
        alpha=0.2,
    )

    plt.xlabel('Age')
    plt.ylabel(brain_region_hemisphere)
    plt.title(f'{brain_region_hemisphere} vs Age for {sex} Participants')
    plt.show()

    for n in range(len(percentiles)):
        quantile = quantiles[n]
        percentile = percentiles[n]

        quantile_column_name = get_brain_region_percentile_cn(percentile, hemisphere_name)
        gpr_params[quantile_column_name] = norm.ppf(quantile, loc=mean, scale=std)

    return pd.DataFrame.from_dict(gpr_params)

