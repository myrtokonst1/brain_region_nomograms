from enum import Enum

from constants.nomogram_constants import amygdala_volume_y_lim
from constants.ukb_table_column_names import r_amygdala_vol_cn, l_amygdala_vol_cn


class AmygdalaVolume(Enum):
    LEFT = 1
    RIGHT = 2

    def __str__(self):
        return self.name.capitalize() + ' Amygdala Volume'

    def get_column_name(self):
        if self.value == 1:
            return l_amygdala_vol_cn
        else:
            return r_amygdala_vol_cn

    def get_name(self):
        return self.name.lower()+'_amygdala_volume'

    @classmethod
    def get_y_lims_nomogram(cls):
        return amygdala_volume_y_lim

    @classmethod
    def get_names(cls):
        return [cls.LEFT.get_name(), cls.RIGHT.get_name()]
