import enum

import pandas as pd
from matplotlib import pyplot as plt

from constants.ukb_table_column_names import cranial_vol_cn, latest_age_cn


def convert_dataset_to_pd(path):
    data = pd.read_csv(path)

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


