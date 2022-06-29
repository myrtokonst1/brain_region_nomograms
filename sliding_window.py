import enum

import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage

from constants.constants import percentiles, quantiles, gaussian_width
from constants.table_columns import latest_age_cn
from utils import generate_list_of_rounded_items, get_percentile_field


def perform_gaussian_smoothing(raw_data: list, gaussian_width=20):
    smoothed_data = ndimage.gaussian_filter1d(raw_data, gaussian_width, mode='nearest')
    return smoothed_data


def perform_sliding_window_analysis(df: pd.DataFrame, brain_region: enum):
    # sort by age
    df = df.sort_values(by=[latest_age_cn])

    number_of_rows = df.shape[0]  # number of participants
    number_of_bins = 100  # a total of 100 bins (we don't end up having 100, we have 91)
    percentage_of_parts_in_bin = 10  # each bin should have 10% of the samples

    # for both the shifts and the number of participants in each bin,
    # we divide the total sample by 100 and 10 respectively
    # since this division will not always be exact, we need to take into account the remainders
    # hence, we create lists in which the inexact division is taken into account
    # eg. if we had 15 participants and wanted 2 bins, the list for parts in a bin would be [8, 7]
    shifts = generate_list_of_rounded_items(number_of_rows, number_of_bins)
    number_of_participants_in_bins = generate_list_of_rounded_items(number_of_rows, percentage_of_parts_in_bin)

    sliding_window_params = {
        'median_ages': [],
        'min_ages': [],
        'max_ages': [],
    }

    for hemisphere in brain_region.get_names():
        sliding_window_params[f'{hemisphere}_average'] = []

        for percentile in percentiles:
            sliding_window_params[get_percentile_field(percentile, hemisphere)] = []

    i = 0
    j = 0
    step = shifts[j]
    while i < number_of_rows - number_of_participants_in_bins[-1]:
        bin = df[i:i+number_of_participants_in_bins[j]]

        sliding_window_params['median_ages'].append(bin[latest_age_cn].median())
        sliding_window_params['min_ages'].append(bin[latest_age_cn].min())
        sliding_window_params['max_ages'].append(bin[latest_age_cn].max())

        for hemisphere in brain_region:
            hemisphere_name = hemisphere.get_name()
            column_name = hemisphere.get_column_name()
            quantiles_for_brain_region = bin[column_name].quantile(quantiles)

            sliding_window_params[f'{hemisphere_name}_average'].append(df[column_name].mean())

            for n in range(len(percentiles)):
                quantile = quantiles[n]
                percentile = percentiles[n]

                quantile_column_name = get_percentile_field(percentile, hemisphere_name)
                sliding_window_params[quantile_column_name].append(quantiles_for_brain_region[quantile])

        i += step
        j += 1
        step = shifts[j]

    for hemisphere in brain_region.get_names():
        for percentile in percentiles:
            quantile_column_name = get_percentile_field(percentile, hemisphere)
            raw_data = sliding_window_params[quantile_column_name]
            sliding_window_params[f'{quantile_column_name}_smoothed'] = perform_gaussian_smoothing(raw_data, gaussian_width)

    return pd.DataFrame.from_dict(sliding_window_params)

