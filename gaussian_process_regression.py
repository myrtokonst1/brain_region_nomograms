import enum
import pathlib
import sys
from collections import Counter

import gpflow
import tensorflow as tf
import matplotlib.pyplot as plt
from gpflow.ci_utils import ci_niter
from gpflow.monitor import Monitor, MonitorTaskGroup, ModelToTensorBoard, ImageToTensorBoard
from gpflow.utilities import print_summary, to_default_float
from scipy.stats import norm

import numpy as np
np.set_printoptions(threshold=sys.maxsize)

import pandas as pd
from scipy.spatial.distance import pdist

from constants.nomogram_constants import percentiles, quantiles
from constants.processed_table_column_names import median_ages, get_brain_region_average_cn, \
    get_brain_region_percentile_cn, get_brain_region_variance_cn
from constants.ukb_table_column_names import latest_age_cn
from enums.brain_regions.hippocampal_volume import HippocampalVolume

number_of_datapoints = 6000
min_age = 40
max_age = 95
number_of_bins = (max_age - min_age)*4  # 4 points per year
median_ages_sequence = np.linspace(min_age, max_age, number_of_bins)


def initialize_params_dictionary(brain_region):
    gpr_params = {
        median_ages: median_ages_sequence.tolist(),
    }

    # initialize average, variance and quantile column names
    # for hemisphere in brain_region.get_names():
    gpr_params[get_brain_region_average_cn(brain_region.get_name())] = []
    gpr_params[get_brain_region_variance_cn(brain_region.get_name())] = []

    for percentile in percentiles:
        gpr_params[get_brain_region_percentile_cn(percentile, brain_region.get_name())] = []

    return gpr_params


def plot_gpr_model(X, Y, xx, mean, std, y_label, title, filename, save=True, plot=True):
    plt.plot(X, Y, "x", mew=2, color='#E1E4EA')
    plt.plot(xx, mean, "C0", lw=2)
    plt.fill_between(
        xx[:, np.newaxis][:, 0],
        mean - 1.96 * std,
        mean + 1.96 * std,
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


def perform_gaussian_process_regression(df: pd.DataFrame, brain_region: enum, sex: enum, random_restarts=10, save=False, plot=False):
    filename = f'GPR_{brain_region.get_name()}_{sex.get_name()}'

    gpr_params = initialize_params_dictionary(brain_region)
    sampled_df = df[:number_of_datapoints]
    # sampled_df = df.sample(n=number_of_datapoints)
    # for brain_region_hemisphere in brain_region:
    # get X and y
    X = sampled_df[latest_age_cn].to_numpy()
    X = X.reshape(-1,1)

    y = sampled_df[brain_region.get_column_name()].to_numpy()
    y = y.reshape(-1,1)
    mean_brain_region_volume = np.mean(X)
    mean_age_difference = np.mean(pdist(y))

    kernel = gpflow.kernels.RBF(variance=mean_brain_region_volume, lengthscales=mean_age_difference)

    model = gpflow.models.GPR(data=(X, y), kernel=kernel)
    print_summary(model)
    log_dir = f'logs/{filename}'

    # awful but has great results in TensorBoard
    def plot_prediction(fig, ax):
        Xnew = np.linspace(X.min() - 0.5, X.max() + 0.5, 100).reshape(-1, 1)
        Ypred = model.predict_f_samples(Xnew, full_cov=True, num_samples=20)
        ax.plot(X, y, "x", mew=2, color='#E1E4EA')
        ax.plot(Xnew.flatten(), np.squeeze(Ypred).T, "C1", alpha=0.2)

    image_task = ImageToTensorBoard(log_dir, plot_prediction, "image_samples")
    model_task = ModelToTensorBoard(log_dir, model)

    slow_tasks = MonitorTaskGroup(image_task, period=5)
    fast_tasks = MonitorTaskGroup(model_task, period=1)
    monitor = Monitor(fast_tasks, slow_tasks)

    for i in range(ci_niter(random_restarts)):
        print(f'iteration {i}')
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=100))
        monitor(i)
    print_summary(model)

    save_dir = str(pathlib.Path(f'saved_models/{filename}'))
    tf.saved_model.save(model, save_dir)

    mean, variance = model.predict_y(median_ages_sequence[:, np.newaxis])
    mean = mean.numpy().flatten()
    std = np.sqrt(variance.numpy()).flatten()

    hemisphere_name = brain_region.get_name()
    gpr_params[get_brain_region_average_cn(hemisphere_name)] = mean.tolist()
    gpr_params[get_brain_region_variance_cn(hemisphere_name)] = std.tolist()

    plot_gpr_model(X, y,median_ages_sequence, mean, std, y_label=brain_region, title=f'{brain_region} vs Age for {sex} Participants', save=save, plot=plot, filename=f'saved_graphs/{filename}.png')

    for n in range(len(percentiles)):
        quantile = quantiles[n]
        percentile = percentiles[n]

        quantile_column_name = get_brain_region_percentile_cn(percentile, hemisphere_name)
        gpr_params[quantile_column_name] = norm.ppf(quantile, loc=mean, scale=std)

    return pd.DataFrame.from_dict(gpr_params)

