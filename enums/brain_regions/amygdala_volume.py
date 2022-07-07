from enum import Enum

from constants.nomogram_constants import amygdala_volume_y_lim
from constants.ukb_table_column_names import r_amygdala_vol_cn, l_amygdala_vol_cn
from enums.brain_regions.brain_region_volume import BrainRegionVolume


class AmygdalaVolume(BrainRegionVolume):
    LEFT = 1
    RIGHT = 2

    def __init__(self, brain_region):
        super().__init__('amygdala volume')

    def get_column_name(self):
        if self.value == 1:
            return l_amygdala_vol_cn
        else:
            return r_amygdala_vol_cn

    @classmethod
    def get_y_lims_nomogram(cls):
        return amygdala_volume_y_lim
