import enum
import string

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from constants.ukb_table_column_names import cranial_vol_cn, latest_age_cn


def convert_dataset_to_pd(path, delim_whitespace=False):
    data = pd.read_csv(path, delim_whitespace=delim_whitespace)

    return data


def scatter_volume_by_age(age, volume, label):
    plt.scatter(age, volume, lw=0, color=[1., 0.5, 0.5],)

    plt.xlabel('Age')
    plt.ylabel(label)
    plt.title(f'Age vs {label}')

    plt.legend()

    plt.show()


def plot_cranial_volume_pre_post_processing(df, healthy_cranial_vol_df):
    plt.scatter(df[latest_age_cn], df[cranial_vol_cn], lw=0, color=[0.2, 0.7, 1.], label='cranial volume')
    plt.scatter(healthy_cranial_vol_df[latest_age_cn], healthy_cranial_vol_df[cranial_vol_cn], lw=0,
                color=[0.9, 0.3, 1.], label='new cranial volume')
    plt.xlabel('Age')
    plt.ylabel('Cranial Volume')
    plt.title('age vs Cranial Volume of all subjects')
    plt.legend()

    plt.show()

    # visualize it in a histogram
    plt.hist(df[cranial_vol_cn], density=True, bins=100, color=[0.2, 0.7, 1.], label='all participants')
    plt.hist(healthy_cranial_vol_df[cranial_vol_cn], density=True, bins=100, color=[0.9, 0.3, 1.],
             label='healthy cranial vols')
    plt.title('Cranial Volume Density')
    plt.xlabel('Cranial Volume')
    plt.ylabel('Density')
    plt.legend()

    plt.show()


# todo rename
def generate_list_of_rounded_items(divident, divisor):
    floored_quotient = int(divident / divisor)
    remainder = divident % divisor
    number_of_floored_elements = divident - remainder
    elements = [floored_quotient] * number_of_floored_elements + [floored_quotient + 1] * remainder

    return elements


def get_percentile_string(percentile: float):
    return str(percentile).replace('.', '_')


def get_underscored_string(phrase: string):
    return '_'.join(phrase.split(' '))


def take_nth_percentile_of_df(df, percentile, sorting_column_name):
    df = df.sort_values(by=[sorting_column_name])
    dataframe_rows = df.shape[0]

    number_of_rows_in_percentile = round(dataframe_rows * percentile)

    bottom_percentile = df.head(number_of_rows_in_percentile)
    top_percentile = df.tail(number_of_rows_in_percentile)

    return bottom_percentile, top_percentile


def compute_difference_between_two_functions(data_1, data_2, y_1_label=None, y_2_label=None, x_label=latest_age_cn, x_values=None, y_average=1):
    difference = np.zeros(data_1.shape[0])
    if x_values is None:
        if data_1[x_label].equals(data_2[x_label]):
            print(f'female average vol {data_1[y_1_label].mean()}')
            print(f'male average vol {data_2[y_2_label].mean()}')
            difference = (data_1[y_1_label] - data_2[y_2_label]).abs()
        else:
            # extrapolate data_2's function on data_1's data
            print('extrapolate data_2s function on data_1s data')
    else:
        # extrapolate both data_2's function and data_1's function on x_data
        print('extrapolate both data_1s function and data_1s function on x_data')

    average_difference = difference.mean()
    relative_average_difference = average_difference/y_average

    return average_difference, relative_average_difference*100