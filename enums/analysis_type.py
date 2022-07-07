import string
from enum import Enum


class AnalysisType(Enum):
    SWA = 'sliding window analysis'
    GPR = 'gaussian process regression'

    def __str__(self):
        return string.capwords(self.value)
