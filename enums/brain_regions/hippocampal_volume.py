from enum import Enum

from constants.nomogram_constants import hippocampal_volume_y_lim
from constants.ukb_table_column_names import l_hippo_vol_cn, r_hippo_vol_cn
from enums.brain_regions.brain_region_volume import BrainRegionVolume


class HippocampalVolume(BrainRegionVolume):
    LEFT = 1
    RIGHT = 2

    def __init__(self, brain_region):
        super().__init__('hippocampal volume')

    def get_column_name(self):
        if self.value == 1:
            return l_hippo_vol_cn
        else:
            return r_hippo_vol_cn

    @classmethod
    def get_y_lims_nomogram(cls):
        return hippocampal_volume_y_lim
