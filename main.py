import pathlib
import shlex

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from analysis.compute_difference_in_nomograms import compute_difference_in_nomograms
from constants.data_paths import get_gpr_filename, data_path
from constants.nomogram_constants import percentiles
from constants.processed_table_column_names import get_brain_region_percentile_cn
from enums.analysis_type import AnalysisType
from prs_regression import perform_prs_regression
from utils import take_nth_percentile_of_df, convert_dataset_to_pd, compute_difference_between_two_functions

pd.set_option('display.max_columns', None)
from enums.brain_regions.amygdala_volume import AmygdalaVolume
from enums.sex import Sex
from gaussian_process_regression import perform_gaussian_process_regression, \
    compute_saved_models_gaussian_process_regression
from nomograms import plot_nomogram, overlay_nomograms
from preprocessing_functions import clean_df, \
    remove_cranial_volume_outliers, correct_brain_region_volume_for_icv_and_scan_date, generate_and_merge_dataframes
from sliding_window import perform_sliding_window_analysis
from constants.ukb_table_column_names import *
from scipy.stats import ttest_ind

from qmplot import manhattanplot

# todo potentially make it a bit more pure
ukb_df = generate_and_merge_dataframes()
ukb_df = clean_df(ukb_df)

# Remove participants whose cranial volume is 5 mean absolute errors from the median
ukb_df = remove_cranial_volume_outliers(ukb_df)

# separate dataframe by sex
is_female = ukb_df[sex_cn] == 0
female_df = ukb_df[is_female]
male_df = ukb_df[~is_female]

# get ENIGMA GWAS data and make the manhattan plot
my_result = pd.read_table(f'{data_path}/amygdala_eur_z_ldsc_restricted_NG05SEP19', '\s+')
my_result['CHR'] = my_result['MarkerName'].apply(lambda x: int(x.split(':')[0]))
my_result['BP'] = my_result['MarkerName'].apply(lambda x: int(x.split(':')[1]))

ax = manhattanplot(data=my_result, chrom="CHR", pos="BP", pv="P", snp="rsid", title='Manhattan plot of ENIGMA amygdala volume GWAS')
plt.show()
plt.savefig("saved_graphs/output_manhattan_plot.png")

brain_region = AmygdalaVolume

for sex in Sex:
    sex_df = female_df if sex == Sex.FEMALE else male_df
    sex.set_dataframe(sex_df)

    for hemisphere in brain_region:
        column_name = hemisphere.get_column_name()
        corrected_volume = correct_brain_region_volume_for_icv_and_scan_date(sex_df, hemisphere, sex, plot=False)
        sex_df.drop(columns=[column_name])
        sex_df[column_name] = corrected_volume

        # Perform Sliding Window Analysis
        sliding_window_bins = perform_sliding_window_analysis(sex_df, hemisphere)
        plot_nomogram(sliding_window_bins, hemisphere, sex, save=True, y_lim=brain_region.get_y_lims_nomogram())

        # Perform Gaussian Process Regression
        # gpr_bins = perform_gaussian_process_regression(sex_df, hemisphere, sex, plot=False)
        # plot_nomogram(sex_df, gpr_bins, hemisphere, sex, save=True, x_lim=[34,100], y_lim=brain_region.get_y_lims_nomogram(),analysis_type=AnalysisType.GPR)

        # fetch the GPR data - you don't have to perform GPR the data is saved
        filename = get_gpr_filename(hemisphere, sex)
        filename_top = get_gpr_filename(hemisphere, sex, extra='high_30')
        filename_bottom = get_gpr_filename(hemisphere, sex, extra='low_30')

        gpr_df = convert_dataset_to_pd(filename)
        df_top = convert_dataset_to_pd(filename_top)
        df_bottom = convert_dataset_to_pd(filename_bottom)

        # Plot nomograms
        plot_nomogram(bins_1=gpr_df, brain_region_1=hemisphere, sex=sex, bin_2=df_top, save=True, x_lim=[45, 82], y_lim=brain_region.get_y_lims_nomogram(), analysis_type=AnalysisType.GPR, p='high')
        plot_nomogram(bins_1=gpr_df, brain_region_1=hemisphere, sex=sex, bin_2=df_bottom, save=True, x_lim=[45, 82], y_lim=brain_region.get_y_lims_nomogram(), analysis_type=AnalysisType.GPR, p='low')

        overlay_nomograms(x_lim=[45, 82], bins_1=sliding_window_bins, type_1=AnalysisType.SWA, sex=sex, bins_2=gpr_df,
                          type_2=AnalysisType.GPR, y_lim=brain_region.get_y_lims_nomogram(), brain_region=hemisphere)

        # Do PRS Analysis and print table
        perform_prs_regression(sex_df, sex, hemisphere)

        # Find the difference between two lines
        compute_difference_in_nomograms(ukb_df, brain_region)

        # Calculate t-test between genetically adjusted and unadjusted nomograms
        fiftieth_percentile_column_name = get_brain_region_percentile_cn(percentiles[4], hemisphere.get_name())
        t_test_low_unadjusted = ttest_ind(gpr_df[fiftieth_percentile_column_name], df_top[fiftieth_percentile_column_name])
        t_test_high_unadjusted = ttest_ind(gpr_df[fiftieth_percentile_column_name], df_bottom[fiftieth_percentile_column_name])
        print(sex, hemisphere)
        print(f't-test high PGS vs unadjusted: {t_test_high_unadjusted}')
        print(f't-test low PGS vs unadjusted: {t_test_low_unadjusted}')