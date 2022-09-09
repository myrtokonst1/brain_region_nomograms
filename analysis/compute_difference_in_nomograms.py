import numpy as np

from constants.data_paths import get_gpr_filename
from constants.nomogram_constants import percentiles
from constants.processed_table_column_names import get_brain_region_percentile_cn
from constants.ukb_table_column_names import sex_cn
from enums.sex import Sex
from preprocessing_functions import correct_brain_region_volume_for_icv_and_scan_date
from utils import convert_dataset_to_pd, compute_difference_between_two_functions


def compute_difference_in_nomograms(ukb_df, brain_region):
    low_diffs = []
    low_rel_diffs = []
    high_diffs = []
    high_rel_diffs = []
    low_high_diffs = []
    low_high_rel_diffs = []
    for sex in Sex:
        is_female = ukb_df[sex_cn] == 0
        sex_df = ukb_df[is_female] if sex == Sex.FEMALE else ukb_df[~is_female]
        for hemisphere in brain_region:
            column_name = hemisphere.get_column_name()
            corrected_volume = correct_brain_region_volume_for_icv_and_scan_date(sex_df, hemisphere, sex, plot=False)
            sex_df.drop(columns=[column_name])
            sex_df[column_name] = corrected_volume

            # fetch the GPR data
            filename = get_gpr_filename(hemisphere, sex)
            filename_top = get_gpr_filename(hemisphere, sex, extra='high_30')
            filename_bottom = get_gpr_filename(hemisphere, sex, extra='low_30')

            gpr_df = convert_dataset_to_pd(filename)
            df_top = convert_dataset_to_pd(filename_top)
            df_bottom = convert_dataset_to_pd(filename_bottom)

            fiftieth_percentile_column_name = get_brain_region_percentile_cn(percentiles[4], hemisphere.get_name())
            amygdala_vol_average = ukb_df[hemisphere.get_column_name()].mean()

            diff, relative_diff= compute_difference_between_two_functions(data_1=gpr_df, data_2=df_bottom, y_1_label=fiftieth_percentile_column_name, y_2_label=fiftieth_percentile_column_name, x_label='median_ages', y_average=amygdala_vol_average)
            print('WHOLE DF VS LOW')
            print(hemisphere, sex)
            print(f'{np.round(diff, decimals=2)} & {np.round(relative_diff, decimals=2)}')
            low_diffs.append(np.round(diff, decimals=2))
            low_rel_diffs.append(np.round(relative_diff, decimals=2))

            diff, relative_diff= compute_difference_between_two_functions(data_1=gpr_df, data_2=df_top, y_1_label=fiftieth_percentile_column_name, y_2_label=fiftieth_percentile_column_name, x_label='median_ages', y_average=amygdala_vol_average)
            print('WHOLE DF VS HIGH')
            print(hemisphere, sex)
            print(f'{np.round(diff, decimals=2)} & {np.round(relative_diff, decimals=2)}')
            high_diffs.append(np.round(diff, decimals=2))
            high_rel_diffs.append(np.round(relative_diff, decimals=2))

            diff, relative_diff= compute_difference_between_two_functions(data_1=df_top, data_2=df_bottom, y_1_label=fiftieth_percentile_column_name, y_2_label=fiftieth_percentile_column_name, x_label='median_ages', y_average=amygdala_vol_average)
            print('HIGH VS LOW')
            print(hemisphere, sex)
            print(f'{np.round(diff, decimals=2)} & {np.round(relative_diff, decimals=2)}')
            low_high_diffs.append(np.round(diff, decimals=2))
            low_high_rel_diffs.append(np.round(relative_diff, decimals=2))

    print('WHOLE DF VS LOW')
    print(np.sum(low_diffs) / 4)
    print(np.sum(low_rel_diffs) / 4)

    print('WHOLE DF VS HIGH')
    print(np.sum(high_diffs) / 4)
    print(np.sum(high_rel_diffs) / 4)

    print('HIGH VS LOW')
    print(np.sum(low_high_diffs) / 4)
    print(np.sum(low_high_rel_diffs) / 4)