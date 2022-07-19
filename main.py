from enums.analysis_type import AnalysisType
from enums.brain_regions.amygdala_volume import AmygdalaVolume
from enums.brain_regions.hippocampal_volume import HippocampalVolume
from enums.sex import Sex
from gaussian_process_regression import perform_gaussian_process_regression
from gaussian_process_regression_scipy import perform_gaussian_process_regression_gpflow
from nomograms import plot_nomogram
from preprocessing_functions import clean_df, \
    remove_cranial_volume_outliers, correct_brain_region_volume_for_icv_and_scan_date, generate_and_merge_dataframes
from sliding_window import perform_sliding_window_analysis
from constants.ukb_table_column_names import *

# todo potentially make it a bit more pure
ukb_df = generate_and_merge_dataframes()
ukb_df = clean_df(ukb_df)

# Remove participants whose cranial volume is 5 mean absolute errors from the median
ukb_df = remove_cranial_volume_outliers(ukb_df)

# separate dataframe by sex
is_female = ukb_df[sex_cn] == 0
female_df = ukb_df[is_female]
male_df = ukb_df[~is_female]

brain_region = HippocampalVolume

male_sliding_window_bins = perform_sliding_window_analysis(male_df, brain_region)
female_sliding_window_bins = perform_sliding_window_analysis(female_df, brain_region)
for sex in Sex:
    sex_df = female_df if sex == Sex.FEMALE else male_df
    sex.set_dataframe(sex_df)

# Correct amygdala volume for ICV
    for volume in brain_region:
        column_name = volume.get_column_name()
        corrected_volume = correct_brain_region_volume_for_icv_and_scan_date(sex_df, volume, sex, plot=False)
        sex_df.drop(columns=[column_name])
        sex_df[column_name] = corrected_volume


# Perform Sliding Window Analysis and plot
#     sliding_window_bins = perform_sliding_window_analysis(sex_df, brain_region)
#     plot_nomogram(sliding_window_bins, brain_region, sex, save=True, y_lim=brain_region.get_y_lims_nomogram())
    # overlay_nomograms(bins_1=sliding_window_bins, brain_region_hemisphere_1=AmygdalaVolume.LEFT, sex_1=sex, bins_2=sliding_window_bins, brain_region_hemisphere_2=AmygdalaVolume.RIGHT, sex_2=sex, y_lim=brain_region.get_y_lims_nomogram())
    gpr_bins = perform_gaussian_process_regression(sex_df, brain_region, sex)
    plot_nomogram(gpr_bins, brain_region, sex, save=True, x_lim=[45,95], y_lim=brain_region.get_y_lims_nomogram(),analysis_type=AnalysisType.GPR)
