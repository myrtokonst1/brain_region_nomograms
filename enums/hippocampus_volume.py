from enum import Enum

from constants.table_columns import l_hippo_vol_cn, r_hippo_vol_cn


class HippocampusVolume(Enum):
    LEFT = 1
    RIGHT = 2

    def __str__(self):
        return self.name.capitalize() + ' Hippocampal Volume'

    def get_column_name(self):
        if self.value == 1:
            return l_hippo_vol_cn
        else:
            return r_hippo_vol_cn

    def get_name(self):
        return self.name.lower()+'_hippocampus_volume'

    @classmethod
    def get_names(cls):
        return [cls.LEFT.get_name(), cls.RIGHT.get_name()]