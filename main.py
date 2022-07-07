from enums.brain_regions.amygdala_volume import AmygdalaVolume
from enums.sex import Sex
from nomograms import plot_nomogram
from preprocessing_functions import clean_df, \
    remove_cranial_volume_outliers, correct_brain_region_volume_for_icv_and_scan_date, generate_and_merge_dataframes
from sliding_window import perform_sliding_window_analysis
from constants.ukb_table_column_names import *

# todo potentially make it a bit more pure
bb_df = generate_and_merge_dataframes()
bb_df = clean_df(bb_df)

# Remove participants whose cranial volume is 5 mean absolute errors from the median
bb_df = remove_cranial_volume_outliers(bb_df)

# separate dataframe by sex
female_df = bb_df[bb_df[sex_cn] == 0]
male_df = bb_df[bb_df[sex_cn] == 1]

brain_region = AmygdalaVolume

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
    sliding_window_bins = perform_sliding_window_analysis(sex_df, brain_region)
    plot_nomogram(sliding_window_bins, brain_region, sex, save=True, y_lim=brain_region.get_y_lims_nomogram())
    # overlay_nomograms(bins_1=sliding_window_bins, brain_region_hemisphere_1=AmygdalaVolume.LEFT, sex_1=sex, bins_2=sliding_window_bins, brain_region_hemisphere_2=AmygdalaVolume.RIGHT, sex_2=sex, y_lim=brain_region.get_y_lims_nomogram())