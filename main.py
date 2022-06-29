import pandas as pd

from enums.amygdala_volume import AmygdalaVolume
from enums.hippocampus_volume import HippocampusVolume
from enums.sex import Sex
from nomograms import plot_nomogram
from preprocessing_functions import clean_df, \
    remove_cranial_volume_outliers, correct_brain_region_volume_for_icv_and_scan_date, generate_and_merge_dataframes
from sliding_window import perform_sliding_window_analysis
from constants.table_columns import *

# todo potentially make it a bit more pure
bb_df = generate_and_merge_dataframes()
bb_df = clean_df(bb_df)

# Remove participants whose cranial volume is 5 mean absolute errors from the median
bb_df = remove_cranial_volume_outliers(bb_df)

# separate dataframe by sex
female_df = bb_df[bb_df[sex_cn] == 0]
male_df = bb_df[bb_df[sex_cn] == 1]

brain_region = HippocampusVolume

# Correct amygdala volume for ICV
for sex in Sex:
    sex_df = female_df if sex == Sex.FEMALE else male_df
    sex.set_dataframe(sex_df)
    for volume in HippocampusVolume:
        column_name = volume.get_column_name()
        corrected_volume = correct_brain_region_volume_for_icv_and_scan_date(sex_df, volume, sex, plot=False)
        sex_df.drop(columns=[column_name])
        sex_df[column_name] = corrected_volume


        # Plot amygdala volume vs age
#         column_name = volume.get_column_name()
#         sex_df = female_df if sex == Sex.FEMALE else male_df
#         sex.set_dataframe(sex_df)
#
#         scatter_volume_by_age(sex.df[age_cn], sex.df[column_name],  f'{str(volume)} {str(sex)} Participants')

    sliding_window_bins = perform_sliding_window_analysis(sex_df, HippocampusVolume)
    plot_nomogram(sliding_window_bins, HippocampusVolume, sex)