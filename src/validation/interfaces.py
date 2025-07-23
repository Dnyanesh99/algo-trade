from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from src.validation.models import DataQualityReport


class IValidator(ABC):
    @abstractmethod
    def validate(self, df: pd.DataFrame, **kwargs: Any) -> tuple[bool, pd.DataFrame, DataQualityReport]:
        pass
