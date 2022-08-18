import matplotlib.pyplot as plt
import numpy as np

from constants.processed_table_column_names import median_ages, \
    get_brain_region_smoothed_percentile_cn, get_brain_region_percentile_cn
from constants.nomogram_constants import percentiles, less_percentiles
from constants.preprocessing_constants import min_age, max_age
from constants.ukb_table_column_names import latest_age_cn
from enums.analysis_type import AnalysisType
from utils import get_underscored_string


def plot_nomogram(bins_1, brain_region_1, sex, x=median_ages, bin_2=None, p='',  x_lim=None, y_lim=None, save=bool, analysis_type=AnalysisType.SWA):
    median_age_column = bins_1[x]
    x_label = ' '.join(x.split('_')).capitalize()

    if not x_lim:
        x_lim = [min_age, max_age]

    for percentile in less_percentiles:
        percentile_column_name = get_brain_region_smoothed_percentile_cn(percentile, brain_region_1.get_name()) if analysis_type == AnalysisType.SWA else get_brain_region_percentile_cn(percentile, brain_region_1.get_name())

        handle_1, = plt.plot(median_age_column, bins_1[percentile_column_name], label=f'Whole dataset Percentile', color='r')

        if bin_2 is not None:
            print(bin_2[percentile_column_name])
            handle_2, = plt.plot(median_age_column, bin_2[percentile_column_name], label=f'{p} Percentile', color='b')

        plt.fill_between(median_age_column,  bin_2[percentile_column_name], bins_1[percentile_column_name], color='#808080', alpha=0.2)

    plt.title(f'{str(brain_region_1)} vs {x_label} for {str(sex)} participants {p} PGS')
    plt.xlabel(x_label)
    plt.ylabel(str(brain_region_1))
    plt.xlim(x_lim)

    if y_lim:
        plt.ylim(y_lim)

    plt.grid(linestyle='--')
    plt.legend(handles=[handle_1, handle_2])

    if save:
        plt.savefig(f'saved_nomograms/{analysis_type.name}_{brain_region_1.get_name()}_nomogram_{sex.get_name()}_normal_vs_{p}.png', dpi=600)

    plt.show()


def overlay_same_type_nomograms(bins_1, brain_region_hemisphere_1, sex_1, bins_2, brain_region_hemisphere_2, sex_2, x=median_ages, x_lim=None, y_lim=None, save=True):
    median_age_column = bins_1[x]
    x_label = ' '.join(x.split('_')).capitalize()

    if not x_lim:
        x_lim = [min_age, max_age]

    for percentile in percentiles:
        percentile_column_name_1 = get_brain_region_smoothed_percentile_cn(percentile, brain_region_hemisphere_1.get_name())
        percentile_column_name_2 = get_brain_region_smoothed_percentile_cn(percentile, brain_region_hemisphere_2.get_name())
        handle_1, = plt.plot(median_age_column, bins_1[percentile_column_name_1], label=f'{brain_region_hemisphere_1} for {sex_1}', color="#CC6677")
        handle_2, = plt.plot(median_age_column, bins_2[percentile_column_name_2], label=f'{brain_region_hemisphere_2} for {sex_2}', color="#888888")

    nomogram_description = ''
    if brain_region_hemisphere_1 == brain_region_hemisphere_2:
        nomogram_description+= f'{str(brain_region_hemisphere_1)} '
    else:
        nomogram_description+= f'{str(brain_region_hemisphere_1)} and {str(brain_region_hemisphere_2)} '

    if sex_1 == sex_2:
        nomogram_description+= f'for {sex_1} participants'
    else:
        nomogram_description+= f'for {sex_1} and {sex_2} participants'

    plt.title(f'Overlay of nomograms of {nomogram_description}')
    # plt.xlabel(x_label)
    plt.ylabel()
    plt.xlim(x_lim)
    if y_lim:
        plt.ylim(y_lim)
    plt.legend(handles=[handle_1, handle_2])
    fig = plt.gcf()
    fig.set_size_inches((8.5, 11), forward=False)
    if save:
        fig.savefig(f'saved_nomograms/{get_underscored_string(nomogram_description)}_nomogram_overlays.png', dpi=600)

    plt.show()


def overlay_nomograms(bins_1, type_1, sex, bins_2, type_2, brain_region, x=median_ages, x_lim=None, y_lim=None, save=True):
    x_label = ' '.join(x.split('_')).capitalize()
    if not x_lim:
        x_lim = [min_age, max_age]

    for percentile in percentiles:
        if type_1 == AnalysisType.SWA:
            handle_2, = plt.plot(bins_2[x], bins_2[get_brain_region_percentile_cn(percentile, brain_region.get_name())], label=f'{type_2}', color="#7D838C")
            handle_1, = plt.plot(bins_1[x], bins_1[get_brain_region_smoothed_percentile_cn(percentile, brain_region.get_name())], label=f'{type_1}', color="#D00B14")
        else:
            handle_1, = plt.plot(bins_2[x], bins_2[get_brain_region_percentile_cn(percentile, brain_region.get_name())], label=f'{type_2}', color="#7D838C")
            handle_2, = plt.plot(bins_1[x], bins_1[get_brain_region_smoothed_percentile_cn(percentile, brain_region.get_name())], label=f'{type_1}', color="#D00B14")

    plt.title(f'Overlay of nomograms of SWA and GPR for {brain_region} for {sex} participants')
    plt.xlabel(x_label)
    plt.ylabel(f'{brain_region}')
    plt.xlim(x_lim)
    if y_lim:
        plt.ylim(y_lim)
    plt.legend(handles=[handle_1, handle_2])
    fig = plt.gcf()
    fig.set_size_inches((8.5, 11), forward=False)
    if save:
        fig.savefig(f'saved_nomograms/{brain_region.get_name()}_{sex}_nomogram_SWA_GPR_overlays_TEST.png', dpi=600)

    plt.show()

