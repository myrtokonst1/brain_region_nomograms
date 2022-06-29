import matplotlib.pyplot as plt

from constants.sliding_window_constants import percentiles
from utils import get_percentile_field


def plot_nomogram(bins, brain_region, sex, x_label='median_ages', x_lim=[45, 82], y_lim=[3000, 5100]):
    for brain_region_hemisphere in brain_region:
        median_age = bins[x_label]
        for percentile in percentiles:
            percentile_column_name = get_percentile_field(percentile, brain_region_hemisphere.get_name())
            plt.plot(median_age, bins[f'{percentile_column_name}_smoothed'], label=f'{percentile} Percentile')

        plt.title(f'{str(brain_region_hemisphere)} vs Median Age for {str(sex)} participants')
        plt.xlabel('Median Ages')
        plt.ylabel(str(brain_region_hemisphere))
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.legend()
        plt.show()