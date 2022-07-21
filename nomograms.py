import matplotlib.pyplot as plt

from constants.processed_table_column_names import median_ages, \
    get_brain_region_smoothed_percentile_cn, get_brain_region_percentile_cn
from constants.nomogram_constants import percentiles
from constants.preprocessing_constants import min_age, max_age
from constants.ukb_table_column_names import latest_age_cn
from enums.analysis_type import AnalysisType
from utils import get_underscored_string


def plot_nomogram(df, bins, brain_region, sex, x=median_ages, x_lim=None, y_lim=None, save=bool, analysis_type=AnalysisType.SWA):
    median_age_column = bins[x]
    x_label = ' '.join(x.split('_')).capitalize()

    if not x_lim:
        x_lim = [min_age, max_age]

    for brain_region_hemisphere in brain_region:
        plt.plot(df[latest_age_cn], df[brain_region_hemisphere.get_column_name()], "kx", mew=2, color='#E1E4EA')
        for percentile in percentiles:
            percentile_column_name = get_brain_region_smoothed_percentile_cn(percentile, brain_region_hemisphere.get_name()) if analysis_type == AnalysisType.SWA else get_brain_region_percentile_cn(percentile, brain_region_hemisphere.get_name())
            plt.plot(median_age_column, bins[percentile_column_name], label=f'{percentile} Percentile', color='r')

        plt.title(f'{str(brain_region_hemisphere)} vs {x_label} for {str(sex)} participants')
        plt.xlabel(x_label)
        plt.ylabel(str(brain_region_hemisphere))
        plt.xlim(x_lim)

        if y_lim:
            plt.ylim(y_lim)

        plt.grid(linestyle='--')

        if save:
            plt.savefig(f'saved_nomograms/{analysis_type.name}_{brain_region_hemisphere.get_name()}_nomogram_{sex.get_name()}.png', dpi=600)

        plt.show()


def overlay_nomograms(bins_1, brain_region_hemisphere_1, sex_1, bins_2, brain_region_hemisphere_2, sex_2, x=median_ages, x_lim=None, y_lim=None, save=True):
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

