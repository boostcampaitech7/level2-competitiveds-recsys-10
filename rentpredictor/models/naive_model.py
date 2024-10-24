import numpy as np
import pandas as pd
from tqdm import tqdm
from .model import Model
from ..preprocessor import Preprocessor

class NaiveModel(Model):
    """직전 N개의 거래를 평균하여 예측하는 모델입니다.
    """

    def preprocess(self, preprocessor: Preprocessor) -> None:
        """지리 관련 feature를 추가합니다.

        Parameters
        ----------
        preprocessor : Preprocessor
            전처리될 데이터를 담고 있는 Preprocessor 객체입니다.
        """
        preprocessor.add_location_id()
        preprocessor.add_location_with_area_id()
        preprocessor.add_contract_datetime()

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
        house_df = pd.concat([X, y], axis=1)
        house_df.sort_values(by=['location_with_area_id', 'contract_datetime'], inplace=True)
        house_df = house_df[['location_with_area_id', 'deposit', 'floor', 'contract_datetime']]
        last_transactions = house_df.groupby('location_with_area_id', observed=True)

        self.target = {}
        self.impute_value = np.nan
        for area_id, group in tqdm(last_transactions):
            group = group.tail(3)
            cv = group['deposit'].std() / group['deposit'].mean()
            if cv > 0.3:
                continue
            self.target[area_id] = group['deposit'].mean()
        
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
        y_pred = X['location_with_area_id'].map(lambda x: self.target.get(x, self.impute_value))
        return y_pred