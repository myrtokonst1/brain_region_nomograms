import enum

from utils import get_percentile_string


scan_timestamp = 'scan_timestamp'
icv_estimate = 'icv_estimate'

median_ages = 'median_ages'
min_ages = 'min_ages'
max_ages = 'max_ages'


def get_brain_region_average_cn(brain_region):
    return f'{brain_region}_average'


def get_brain_region_variance_cn(brain_region):
    return f'{brain_region}_variance'


def get_brain_region_percentile_cn(percentile: float, brain_region: enum):
    quantile_str = get_percentile_string(percentile)
    column_name = f'{brain_region}_{quantile_str}'

    return column_name


def get_brain_region_smoothed_percentile_cn(percentile: float, brain_region: enum):
    return f'{get_brain_region_percentile_cn(percentile,brain_region)}_smoothed'
