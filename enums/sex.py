from enum import Enum

from pandas import DataFrame


class Sex(Enum):
    FEMALE = 1
    MALE = 2

    def __init__(self, df=DataFrame):
        self.df = df

    def __str__(self):
        return self.name.capitalize()

    def get_name(self):
        return self.name.lower()

    def set_dataframe(self, df):
        self.df = df
