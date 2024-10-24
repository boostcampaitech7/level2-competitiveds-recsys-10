import pandas as pd
from tqdm import tqdm
import lightgbm as lgb
from .model import Model
from .geo_model import GeoModel
from .naive_model import NaiveModel
from ..preprocessor import Preprocessor

class EnsembleModel(Model):
    """각종 지리정보가 feature로 추가된 모델입니다.
    """

    def preprocess(self, preprocessor: Preprocessor) -> None:
        """지리 관련 feature를 추가합니다.

        Parameters
        ----------
        preprocessor : Preprocessor
            전처리될 데이터를 담고 있는 Preprocessor 객체입니다.
        """
        GeoModel.preprocess(self, preprocessor)
        NaiveModel.preprocess(self, preprocessor)


    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None) -> None:
        """모델을 학습합니다.

        Parameters
        ----------
        X : pd.DataFrame
            학습 데이터입니다.
        y : pd.Series
            학습 데이터의 target 값입니다.
        X_val : pd.DataFrame, optional
            검증 데이터입니다.
            early stopping을 위한 것으로, 선택적으로 제공합니다.
        y_val : pd.Series, optional
            검증 데이터의 target 값입니다.
            early stopping을 위한 것으로, 선택적으로 제공합니다.
        """
        NaiveModel.fit(self, X, y, X_val, y_val)
        GeoModel.fit(self, X, y, X_val, y_val)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """테스트 데이터에 대해 예측을 수행합니다.

        메서드를 호출하기 전, fit 메서드를 먼저 호출해야합니다.

        Parameters
        ----------
        X : pd.DataFrame
            예측을 수행할 테스트데이터입니다.

        Returns
        -------
        pd.Series
            테스트데이터에 대한 예측한 결과입니다.
        """
        naive_pred = NaiveModel.predict(self, X)
        geo_pred = GeoModel.predict(self, X)

        naive_pred.reset_index(drop=True, inplace=True)
        geo_pred.reset_index(drop=True, inplace=True)

        naive_pred.fillna(geo_pred, inplace=True)
        naive_pred = naive_pred * 0.5 + geo_pred * 0.5
        return naive_pred
