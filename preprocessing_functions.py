import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from constants.data_paths import main_ukb_data_path, ukb_brain_data_path, ukb_preprocessed_data_path, \
    ukb_participant_exclusion_data_path
from constants.preprocessing_constants import white_british_code, min_age, max_age, standard_brain_icv
from constants.processed_table_column_names import scan_timestamp, icv_estimate
from constants.ukb_table_column_names import intercranial_ratio_cn, scan_date_cn, \
    ethnicity_cn, \
    cranial_vol_cn, l_amygdala_vol_cn, r_amygdala_vol_cn, latest_age_cn, icd_cns, exclusion_columns_to_excluded_values, \
    participant_id
from enums.brain_regions.brain_region_volume import BrainRegionVolume
from utils import plot_cranial_volume_pre_post_processing, convert_dataset_to_pd


def generate_and_merge_dataframes():
    # import data and convert it to DataFrame
    main_ukb_df = convert_dataset_to_pd(main_ukb_data_path)
    brain_ukb_df = convert_dataset_to_pd(ukb_brain_data_path)
    preprocessed_ukb_df = convert_dataset_to_pd(ukb_preprocessed_data_path)
    participant_exclusion_ukb_df = convert_dataset_to_pd(ukb_participant_exclusion_data_path)

    # get columns of participant_exclusion_ukb_df that we want to include in the merged dataframe
    desired_participant_exclusion_columns = [column_name for column_name in participant_exclusion_ukb_df.columns.tolist() if any(column_subname in column_name for column_subname in icd_cns)] + [participant_id]

    # merge the three datasets
    bb_df = pd.merge(pd.merge(pd.merge(main_ukb_df, brain_ukb_df[[participant_id, l_amygdala_vol_cn, r_amygdala_vol_cn]], on=participant_id, how='outer'),
                     preprocessed_ukb_df[[participant_id, latest_age_cn]], on=participant_id, how='outer'),
                     participant_exclusion_ukb_df[desired_participant_exclusion_columns], on=participant_id, how='outer')

    return bb_df


def exclude_values_from_column(df, column_name, column_values_to_exclude):
    columns_with_values_to_exclude = [column for column in df.columns.tolist() if any(column_name_subset in column for column_name_subset in [column_name])]

    for column in columns_with_values_to_exclude:
        df = df[~df[column].isin(column_values_to_exclude)]

    return df


def clean_df(df):
    df = df[df[intercranial_ratio_cn].notna()]  # remove patients that don't have amygdala volume values
    df = df[df[l_amygdala_vol_cn].notna()]  # remove patients that don't have amygdala volume values
    df = df[df[r_amygdala_vol_cn].notna()]  # remove patients that don't have amygdala volume values
    df = df[df[ethnicity_cn] == white_british_code]  # only select participants of white british background
    df = df[df[latest_age_cn].between(min_age, max_age)]  # only take participants between the ages of 44-82

    # remove participants that suffer from specific illnesses
    for column_name, column_values_to_exclude in exclusion_columns_to_excluded_values.items():
        df = exclude_values_from_column(df, column_name, column_values_to_exclude)

    return df


# Use regression to remove participants whose cranial volume is more than n maes away from the median
def remove_cranial_volume_outliers(df, plot=False, n=5):
    cranial_vol = df[cranial_vol_cn]
    cranial_vol_median = cranial_vol.median()
    participant_count = df.shape[0]

    mean_absolute_error = np.sum(abs(cranial_vol - cranial_vol_median)) / participant_count
    healthy_cranial_vol_df = df[abs(cranial_vol - cranial_vol_median) < n * mean_absolute_error]

    if plot:
        plot_cranial_volume_pre_post_processing(df, healthy_cranial_vol_df)

    return healthy_cranial_vol_df


# Regress out (ie control for) ICV and scan date
def correct_brain_region_volume_for_icv_and_scan_date(df, brain_region_enum: BrainRegionVolume, sex, plot=False):
    brain_region_column_name = brain_region_enum.get_column_name()

    df[scan_timestamp] = pd.to_datetime(df[scan_date_cn]).map(pd.Timestamp.timestamp)
    df[icv_estimate] = df[intercranial_ratio_cn].map(lambda x: standard_brain_icv / x)

    X = df[[icv_estimate, scan_timestamp]]
    Y = df[brain_region_column_name]

    linear_regressor = LinearRegression()
    linear_regressor.fit(X, Y)

    Y_pred = linear_regressor.predict(X)
    residuals = Y - Y_pred

    # the coefficient with index 0 is taken for ICV since it correlates to the first feature in X
    # equivalently, scan_timestamp is of index 1 since it is the second element in X
    # python is indexed at 0
    ICV_correction = linear_regressor.coef_[0] * X[icv_estimate].mean()
    scan_date_correction = linear_regressor.coef_[1] * X[scan_timestamp].mean()
    corrected_volume = linear_regressor.intercept_ + residuals + ICV_correction + scan_date_correction #  brain region volume for controlling ICV + scan date

    if plot:
        plt.scatter(X[icv_estimate], Y, label=f'Raw {str(brain_region_enum)}')
        plt.scatter(X[icv_estimate], corrected_volume, label=f'Corrected {str(brain_region_enum)} for ICV and scan datetime')
        plt.xlabel('ICV')
        plt.ylabel(str(brain_region_enum))
        plt.title(f'{str(brain_region_enum)} vs ICV before and after correction ({str(sex)})')
        plt.legend()
        plt.show()

        plt.scatter(df[latest_age_cn], df[brain_region_column_name], label=f'Raw {str(brain_region_enum)}')
        plt.scatter(df[latest_age_cn], corrected_volume, label=f'Corrected {str(brain_region_enum)}')
        plt.xlabel('Age')
        plt.ylabel(str(brain_region_enum))
        plt.title(f'{str(brain_region_enum)} vs Age before and after correction for ICV ({str(sex)})')
        plt.legend()
        plt.show()

    return corrected_volume
