import string
from enum import Enum

from constants.nomogram_constants import amygdala_volume_y_lim
from constants.ukb_table_column_names import r_amygdala_vol_cn, l_amygdala_vol_cn
from utils import get_underscored_string


class BrainRegionVolume(Enum):
    # LEFT = 1
    # RIGHT = 2

    def __init__(self, brain_region):
        self.brain_region = brain_region
        print(brain_region.capitalize())

    def __str__(self):
        return self.name.capitalize() + f' {string.capwords(self.brain_region)}'

    def get_column_name(self):
        pass

    def get_name(self):
        return self.name.lower()+f'_{get_underscored_string(self.brain_region)}'

    @classmethod
    def get_y_lims_nomogram(cls):
        return []

    @classmethod
    def get_names(cls):
        return [cls.LEFT.get_name(), cls.RIGHT.get_name()]
