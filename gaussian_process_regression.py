import enum

import GPy as GPy
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.stats import norm

import numpy as np
import pandas as pd
import scipy


# sigma initialised as the average volume of the brain region in question
# L initialised as average age difference between all samples
from scipy.spatial.distance import pdist

from constants.nomogram_constants import percentiles, quantiles
from constants.processed_table_column_names import median_ages, min_ages, max_ages, get_brain_region_average_cn, \
    get_brain_region_percentile_cn, get_brain_region_variance_cn
from constants.ukb_table_column_names import latest_age_cn

number_of_bins = 500
min_age = 30
max_age = 100


def perform_gaussian_process_regression(df:pd.DataFrame, brain_region: enum, sex: enum, save=False, plot=False):
    # GPy.plotting.change_plotting_library('plotly')

    gpr_params = {
        # todo im confused
        median_ages: np.arange(min_age, max_age, (max_age - min_age)/number_of_bins).tolist(),
    }

    for hemisphere in brain_region.get_names():
        gpr_params[get_brain_region_average_cn(hemisphere)] = []
        gpr_params[get_brain_region_variance_cn(hemisphere)] = []

        for percentile in percentiles:
            gpr_params[get_brain_region_percentile_cn(percentile, hemisphere)] = []

    for brain_region_hemisphere in brain_region:
        hemisphere_name = brain_region_hemisphere.get_name()

        X = df[latest_age_cn].iloc[:1000]
        X = X[:, None]

        y = df[brain_region_hemisphere.get_column_name()].iloc[:1000]
        y = y[:, None]

        mean_brain_region_volume = X.mean()
        mean_age_difference = (pdist(y)).mean()

        kernel = GPy.kern.RBF(input_dim=1, variance=mean_brain_region_volume, lengthscale=mean_age_difference)

        model = GPy.models.GPRegression(X, y, kernel)
        display(model)

        if plot:
            model.plot()
            plt.xlabel('Age')
            plt.ylabel(f'{brain_region_hemisphere}')
            plt.title(f'{brain_region_hemisphere} vs Age Pre Gaussian Process Optimization for {sex} Participants')

            if save:
                plt.savefig(
                    f'saved_nomograms/prepocess_GPR_{brain_region_hemisphere.get_name()}_{sex.get_name()}.png',
                    dpi=600)

            plt.show()

        model.optimize(messages=True)
        model.optimize_restarts(num_restarts=10)

        display(model)

        if plot:
            model.plot()
            plt.xlabel('Age')
            plt.ylabel(f'{brain_region_hemisphere}')
            plt.title(f'{brain_region_hemisphere} vs Age Post Gaussian Process Optimization for {sex} Participants')

            if save:
                plt.savefig(
                    f'saved_nomograms/post_pocess_GPR_{brain_region_hemisphere.get_name()}_{sex.get_name()}.png',
                    dpi=600)

            plt.show()

        mean, variance = model.predict(np.linspace(min_age, max_age, number_of_bins)[:, np.newaxis])
        mean = mean.flatten().tolist()

        std = [np.sqrt(v) for v in variance.flatten()]
        gpr_params[get_brain_region_average_cn(hemisphere_name)] = mean
        gpr_params[get_brain_region_variance_cn(hemisphere_name)] = std

        for n in range(len(percentiles)):
            quantile = quantiles[n]
            percentile = percentiles[n]

            quantile_column_name = get_brain_region_percentile_cn(percentile, hemisphere_name)
            gpr_params[quantile_column_name] = norm.ppf(quantile, loc=mean, scale=std)

    return pd.DataFrame.from_dict(gpr_params)

