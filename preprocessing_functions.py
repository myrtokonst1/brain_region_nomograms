import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from constants.constants import main_ukb_data_path, ukb_brain_data_path, ukb_preprocessed_data_path
from constants.table_columns import intercranial_ratio_cn, latest_age_cn, scan_date_cn, standard_brain_icv, \
    ethnicity_cn, \
    white_british_code, cranial_vol_cn, l_amygdala_vol_cn, r_amygdala_vol_cn
from utils import plot_cranial_volume_pre_post_processing, convert_dataset_to_pd


def generate_and_merge_dataframes():
    # import data and convert it to DataFrame
    main_ukb_df = convert_dataset_to_pd(main_ukb_data_path)
    brain_ukb_df = convert_dataset_to_pd(ukb_brain_data_path)
    ukb_preprocessed_df = convert_dataset_to_pd(ukb_preprocessed_data_path)

    bb_df = pd.merge(pd.merge(main_ukb_df, brain_ukb_df, on='eid', how='outer'),
                     ukb_preprocessed_df[['eid', latest_age_cn]], on='eid', how='outer', suffixes=('', '_DROP'))

    return bb_df


def clean_df(df):
    df = df[df[intercranial_ratio_cn].notna()]  # remove patients that don't have amygdala volume values
    df = df[df[l_amygdala_vol_cn].notna()]  # remove patients that don't have amygdala volume values
    df = df[df[r_amygdala_vol_cn].notna()]  # remove patients that don't have amygdala volume values
    df = df[df[ethnicity_cn] == white_british_code]  # only select participants of white british background
    df = df[df[latest_age_cn].between(44, 82)]  # only take participants between the ages of 44-82

    # exclude participants with neuro/psych disorders, head trauma, substance abuse, and cardiovascular disorders
    # todo
    return df


def remove_cranial_volume_outliers(df, plot=False):
    cranial_vol = df[cranial_vol_cn]
    cranial_vol_median = cranial_vol.median()
    participant_count = df.shape[0]

    mean_absolute_error = np.sum(abs(cranial_vol - cranial_vol_median)) / participant_count
    healthy_cranial_vol_df = df[abs(cranial_vol - cranial_vol_median) < 5 * mean_absolute_error]

    if plot:
        plot_cranial_volume_pre_post_processing(df, healthy_cranial_vol_df)

    return healthy_cranial_vol_df


def correct_brain_region_volume_for_icv_and_scan_date(df, brain_region_enum, sex, plot=False):
    column_name = brain_region_enum.get_column_name()

    df['scan_timestamp'] = pd.to_datetime(df[scan_date_cn]).map(pd.Timestamp.timestamp)
    df['icv_estimate'] = df[intercranial_ratio_cn].map(lambda x: standard_brain_icv / x)

    X = df[[latest_age_cn, 'icv_estimate', 'scan_timestamp']]
    Y = df[column_name]

    linear_regressor = LinearRegression()
    linear_regressor.fit(X, Y)

    Y_pred = linear_regressor.predict(X)
    residuals = (Y_pred - Y)

    # the coefficient with index 1 is taken for ICV since it correlates to the second feature in X
    # equivalently, scan_timestamp is of index 2 since it is the third element in X
    # python is indexed at 0
    ICV_correction = linear_regressor.coef_[1] * df['icv_estimate'].mean()
    scan_date_correction = linear_regressor.coef_[2] * df['scan_timestamp'].mean()
    corrected_volume = linear_regressor.intercept_ + residuals + ICV_correction + scan_date_correction    # LHV for controlling ICV + scan date

    if plot:
        plt.scatter(X['icv_estimate'], Y, label=f'Raw {str(brain_region_enum)}')
        plt.scatter(X['icv_estimate'], corrected_volume, label=f'Corrected {str(brain_region_enum)} for ICV')
        plt.xlabel('ICV')
        plt.ylabel(str(brain_region_enum))
        plt.title(f'{str(brain_region_enum)} vs ICV before and after correction ({str(sex)})')
        plt.legend()
        plt.show()

        plt.scatter(df[latest_age_cn], df[column_name], label=f'Raw {str(brain_region_enum)}')
        plt.scatter(df[latest_age_cn], corrected_volume, label=f'Corrected {str(brain_region_enum)}')
        plt.xlabel('Age')
        plt.ylabel(str(brain_region_enum))
        plt.title(f'{str(brain_region_enum)} vs Age before and after correction for ICV ({str(sex)})')
        plt.legend()
        plt.show()

    return corrected_volume
