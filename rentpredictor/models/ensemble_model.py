import copy
import pandas as pd
from .. import utils
from .model import Model
from .geo_model import GeoModel
from .naive_model import NaiveModel

class EnsembleModel(Model):
    """GeoModel과 NaiveModel을 앙상블하여 예측하는 모델입니다.
    """

    def __init__(self) -> None:
        self.geo_model = GeoModel()
        self.naive_model = NaiveModel()

    def set_data(self, dataframes: dict[str, pd.DataFrame]) -> None:
        """데이터를 설정합니다.

        Parameters
        ----------
        dataframes : dict[str, pd.DataFrame]
            학습 및 테스트 데이터, 그리고 그 외 정보를 포함한 딕셔너리입니다.
        """
        self.dataframes = dataframes
        self.geo_model.set_data(copy.deepcopy(dataframes))
        self.naive_model.set_data(copy.deepcopy(dataframes))

    def preprocess(self) -> None:
        """GeoModel과 NaiveModel의 전처리를 수행합니다.
        """
        self.geo_model.preprocess()
        self.naive_model.preprocess()

    def fit(self) -> None:
        """GeoModel과 NaiveModel을 학습합니다.
        """
        self.geo_model.fit()
        self.naive_model.fit()

    def predict(self) -> pd.Series:
        """GeoModel과 NaiveModel을 이용하여 예측을 수행합니다.
        """
        naive_pred = self.naive_model.predict()
        geo_pred = self.geo_model.predict()

        naive_pred.reset_index(drop=True, inplace=True)
        geo_pred.reset_index(drop=True, inplace=True)
        naive_pred.fillna(geo_pred, inplace=True)

        final_pred = 0.4 * naive_pred + 0.6 * geo_pred
        return final_pred
